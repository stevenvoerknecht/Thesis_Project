from transformers import pipeline

# Load the classification pipeline straight from the hub
classifier = pipeline("text-classification", model="Stevenvoerknecht/thesis-champion-model")

# Run inference on text data
sample_text = "Sample text to be analyzed"
results = classifier(sample_text)

print(results)