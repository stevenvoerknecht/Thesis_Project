import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Download and load tokenizer + model weights
model_name = "Stevenvoerknecht/thesis-champion-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize input text strings safely
text = "Sample text to be analyzed"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

# Pass inputs to model for evaluation inference
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits).squeeze(0) 

# Iterate through every label score in the output vector
classifications = {}
for idx, score in enumerate(probabilities):
    score_val = score.item()
    label_name = model.config.id2label[idx]
    classifications[label_name] = round(score_val, 4)

# Output results
if classifications:
    print("Detected Narratives:")
    for label, conf in classifications.items():
        print(f" - {label}: {conf}")
else:
    print("Detected Narratives: None (Clean/Neutral)")