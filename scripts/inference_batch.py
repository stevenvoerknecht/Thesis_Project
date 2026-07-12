import os
import torch
import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Congiguaration constants
INPUT_PARQUET = "data/processed/test_split.pqt"
OUTPUT_PARQUET = "data/inference/test_inference_champion.pqt"
# Hugging Face model: "Stevenvoerknecht/thesis-champion-model"
MODEL_NAME = "Stevenvoerknecht/thesis-champion-model"

BATCH_SIZE = 64        # Adjust based on your GPU VRAM limits
MAX_LENGTH = 512       # Sequence length limit for tokenization
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Narrative-specific optimal decision thresholds
THRESHOLDS = {
    "Populist_narrative": 0.32,
    "Nativist_narrative": 0.44,
    "Denialist_narrative": 0.18,
    "Declinist_narrative": 0.22,
    "Apocalypticist_narrative": 0.3,
    "Revisionist_narrative": 0.26
}

print(f"Using device: {DEVICE.upper()}")

# initialization
print(f"Loading tokenizer and model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Extract labels directly from your model's config metadata
id2label = model.config.id2label
narrative_labels = [id2label[i] for i in sorted(id2label.keys())]
print(f"Loaded target classification dimensions: {narrative_labels}")

# data loading
print(f"Scanning input Parquet files from: {INPUT_PARQUET}")
lf = pl.scan_parquet(INPUT_PARQUET)

# Apply baseline filtering (skipping missing messages or system actions)
df_all = lf.filter(
    pl.col("message_text").is_not_null() & 
    (pl.col("message_text") != "") & 
    pl.col("is_action_type").is_null()
).collect()

messages = df_all["message_text"].to_list()
total_records = len(messages)
print(f"Ready to process {total_records} message rows.")

# batch inference loop
print("Starting batch multi-label evaluation tracking")
all_scores = []

with torch.no_grad():
    for i in tqdm(range(0, total_records, BATCH_SIZE), desc="Inference Batches"):
        batch_texts = messages[i:i + BATCH_SIZE]
        
        # Tokenize batch with defensive slicing/padding
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=MAX_LENGTH, 
            return_tensors="pt"
        ).to(DEVICE)
        
        # Forward pass through sequence classification head
        outputs = model(**inputs)
        
        # Convert logits via Sigmoid for independent multilabel margins
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        all_scores.extend(probs)

# parsing
print("Assembling prediction distributions back to dataframe layout...")

# Dynamically construct evaluation array columns for each label
prediction_columns = []
for idx in sorted(id2label.keys()):
    label_name = id2label[idx]
    label_scores = [row[idx] for row in all_scores]
    
    # Retrieve the class-specific threshold (default to 0.5 if key not found)
    threshold = THRESHOLDS.get(label_name, 0.50)
    
    # Save raw continuous probability score and class-specific binary prediction
    prediction_columns.append(pl.Series(f"{label_name}_score", label_scores))
    prediction_columns.append(pl.Series(f"{label_name}_active", [score >= threshold for score in label_scores]))

# Attach array updates into baseline dataframe
df_final = df_all.with_columns(prediction_columns)

# Construct a clean tracking summary condition for complete target neutrals
active_bool_cols = [f"{label_name}_active" for label_name in narrative_labels]
any_narrative_hit = pl.any_horizontal([pl.col(col) for col in active_bool_cols])

df_final = df_final.with_columns([
    (~any_narrative_hit).alias("no_contested_narrative_present")
])

# export
os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
df_final.write_parquet(OUTPUT_PARQUET)
print(f"Success! Finished classification. Output saved to: {OUTPUT_PARQUET}")