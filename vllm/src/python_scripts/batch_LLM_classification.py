import json
import os
import polars as pl
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Configuration Constants
INPUT_PARQUET = "data/raw/nl_all.pqt"
OUTPUT_PARQUET = "data/processed/labeled_subset.pqt"
PROMPT_FILE = "vllm/src/python_scripts/LLM_prompts/LLM_prompt_v4.md"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAMPLE_SIZE = 50000

MIN_CHAR_LENGTH = 80  
MIN_WORD_COUNT = 12   

if not os.path.exists(PROMPT_FILE):
    raise FileNotFoundError(f"Could not find your prompt file at: {PROMPT_FILE}")

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)

print("Loading Parquet data and generating high-context representative sample")

# Initialize the LazyFrame using the exact schema
lf = pl.scan_parquet(INPUT_PARQUET)

print(f"Generating high-context global random sample of {SAMPLE_SIZE} messages...")

# Global random sampling with strict length and metadata filters
df_sample = (
    lf.filter(
        pl.col("message_text").is_not_null() & 
        (pl.col("message_text") != "") & 
        (pl.col("is_action_type").is_null()) &
        (pl.col("message_text").str.len_chars() >= MIN_CHAR_LENGTH) &
        (pl.col("message_text").str.split(" ").list.len() >= MIN_WORD_COUNT) 
    )
    .collect()
    .sample(n=SAMPLE_SIZE, with_replacement=False, seed=42) 
)

print(f"High-context representative sample created with {len(df_sample)} rows.")

# Define explicit JSON schema blueprint constraint mapping for vLLM guided decoding
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "no_contested_narrative_present": {"type": "boolean"},
        "classifications": {
            "type": "object",
            "properties": {
                "elite_vs_mass_conflict": {"type": "integer", "minimum": 0, "maximum": 3},
                "in_group_vs_out_group_exclusion": {"type": "integer", "minimum": 0, "maximum": 3},
                "institutional_knowledge_denial": {"type": "integer", "minimum": 0, "maximum": 3},
                "societal_moral_regression": {"type": "integer", "minimum": 0, "maximum": 3},
                "imminent_acute_crisis_panic": {"type": "integer", "minimum": 0, "maximum": 3},
                "systemic_sovereignty_revival": {"type": "integer", "minimum": 0, "maximum": 3}
            },
            "required": [
                "elite_vs_mass_conflict", "in_group_vs_out_group_exclusion", 
                "institutional_knowledge_denial", "societal_moral_regression", 
                "imminent_acute_crisis_panic", "systemic_sovereignty_revival"
            ]
        }
    },
    "required": ["rationale", "no_contested_narrative_present", "classifications"]
}

print("Initializing vLLM Offline Engine...")
llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=1, 
    max_model_len=4096,
    trust_remote_code=True,
    enforce_eager=True
)

# Set generation parameter bounds
sampling_params = SamplingParams(
    temperature=0.0, 
    max_tokens=1024,
)

# Inject schema constraints directly into the runtime data config dict
sampling_params.guided_decoding_data = {"json": json.dumps(JSON_SCHEMA)}

print("Formatting prompts using the native model template...")
messages = df_sample["message_text"].to_list()

tokenizer = llm.get_tokenizer()

# The model allows max 4096 tokens including system prompt
MAX_ALLOWED_USER_TOKENS = 2000

formatted_prompts = []
for msg in messages:
    msg_token_ids = tokenizer.encode(msg, add_special_tokens=False)
    
    if len(msg_token_ids) > MAX_ALLOWED_USER_TOKENS:
        msg_token_ids = msg_token_ids[:MAX_ALLOWED_USER_TOKENS]
        truncated_msg = tokenizer.decode(msg_token_ids, skip_special_tokens=True)
    else:
        truncated_msg = msg
        
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this message:\n\n{truncated_msg}"}
    ]
    
    rendered_string = tokenizer.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=True
    )
    formatted_prompts.append(rendered_string)

print(f"Running batch offline inference on {len(messages)} rows")
outputs = llm.generate(
    prompts=formatted_prompts, 
    sampling_params=sampling_params
)
raw_json_responses = [output.outputs[0].text for output in outputs]

print("Parsing structural JSON arrays using fault-tolerant Python mappings")

df_results = df_sample.with_columns([
    pl.Series("raw_llm_output", raw_json_responses)
])

def safe_parse_json(raw_str):
    try:
        return json.loads(raw_str)
    except Exception:
        return None

json_struct_schema = pl.Struct([
    pl.Field("rationale", pl.String),
    pl.Field("no_contested_narrative_present", pl.Boolean),
    pl.Field("classifications", pl.Struct([
        pl.Field("elite_vs_mass_conflict", pl.Int64),
        pl.Field("in_group_vs_out_group_exclusion", pl.Int64),
        pl.Field("institutional_knowledge_denial", pl.Int64),
        pl.Field("societal_moral_regression", pl.Int64),
        pl.Field("imminent_acute_crisis_panic", pl.Int64),
        pl.Field("systemic_sovereignty_revival", pl.Int64)
    ]))
])

# Safely map elements and unnest root properties
df_final = df_results.with_columns([
    pl.col("raw_llm_output").map_elements(
        safe_parse_json, 
        return_dtype=json_struct_schema
    ).alias("parsed_json")
]).unnest("parsed_json")

# Safely unnest nested classification metrics
if "classifications" in df_final.columns:
    df_final = df_final.unnest("classifications")

# List of tracking dimensions for structural integrity iteration checks
narrative_cols = [
    "elite_vs_mass_conflict", "in_group_vs_out_group_exclusion", 
    "institutional_knowledge_denial", "societal_moral_regression", 
    "imminent_acute_crisis_panic", "systemic_sovereignty_revival"
]

# Fill parsing nulls with 0s so boolean conditional maps do not crash
df_final = df_final.with_columns([
    pl.col(col).fill_null(0) for col in narrative_cols
])

# Expression tracking active narrative items
any_narrative_detected = pl.any_horizontal([pl.col(col) > 0 for col in narrative_cols])

# If any metrics > 0, override "no contested narratives" to False
df_final = df_final.with_columns([
    pl.when(any_narrative_detected)
    .then(False)
    .otherwise(pl.col("no_contested_narrative_present"))
    .alias("no_contested_narrative_present")
])

if "raw_llm_output" in df_final.columns:
    df_final = df_final.drop("raw_llm_output")

failed_count = df_final["rationale"].is_null().sum()
if failed_count > 0:
    print(f"Cleaned up {failed_count} truncated/malformed JSON strings (set to null).")

df_final.write_parquet(OUTPUT_PARQUET)
print(f"Success! Classified dataset saved cleanly to {OUTPUT_PARQUET}")