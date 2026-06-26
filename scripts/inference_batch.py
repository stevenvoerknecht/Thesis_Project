import os
import torch
import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Congiguaration constants
INPUT_PARQUET = "data/raw/<YOUR_FILE_NAME>.pqt"
OUTPUT_PARQUET = "data/processed_predictions/final_labeled.pqt"
MODEL_NAME = "Stevenvoerknecht/thesis-champion-model"

BATCH_SIZE = 64        # Adjust based on your GPU VRAM limits
MAX_LENGTH = 512       # Sequence length limit for tokenization
THRESHOLD = 0.15       # Confidence boundary for active narratives
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
for idx, label_name in id2label.items():
    # Gather array scores for this specific column position index
    label_scores = [row[idx] for row in all_scores]
    
    # Create two columns per label: raw probability float, and active binary boolean flag
    prediction_columns.append(pl.Series(f"{label_name}_score", label_scores))
    prediction_columns.append(pl.Series(f"{label_name}_active", [score >= THRESHOLD for score in label_scores]))

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