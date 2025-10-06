from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

MODEL_PATH = "models/fine_tuned_classifier"

# Only run conversion if safetensors doesn't exist
safetensors_path = os.path.join(MODEL_PATH, "model.safetensors")
if not os.path.exists(safetensors_path):
    print("Converting to safetensors...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.save_pretrained(MODEL_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MODEL_PATH)
    print("Conversion complete.")
else:
    print("Safetensors already exist, skipping conversion.")