# check_model_labels_peft.py
import json
from transformers import AutoModelForSequenceClassification
from peft import PeftModel


BASE_MODEL = "microsoft/deberta-v3-large"
PEFT_MODEL_PATH = "models/fine_tuned_classifier"  # LoRA adapter path
TRAINING_DATA = "data/ground_truth.jsonl"


def check_peft_model_labels(base_model_name, peft_path):
    """Load base model + LoRA adapter and print label info"""
    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
        )
        model = PeftModel.from_pretrained(base_model, peft_path)

        num_labels = model.base_model.config.num_labels
        print(f"PEFT model loaded from: {peft_path}")
        print(f"Number of labels in base model: {num_labels}")

        if hasattr(model.base_model.config, "id2label") and model.base_model.config.id2label:
            print(f"Labels found: {len(model.base_model.config.id2label)}")
            print("First few labels:")
            for i, label in list(model.base_model.config.id2label.items())[:5]:
                print(f"  {i}: {label}")
            if len(model.base_model.config.id2label) > 5:
                print("...")
        else:
            print("No label mapping found in base model config")

        return num_labels
    except Exception as e:
        print(f"Error loading PEFT model: {e}")
        return None


def count_training_labels(training_file):
    """Count unique combined labels from training data"""
    labels = set()
    try:
        with open(training_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                combined = "|".join([
                    entry["task_type"],
                    entry["output_modality"],
                    entry["complexity"],
                    entry["temporal_context"],
                    entry["spatial_context"]
                ])
                labels.add(combined)

        print(f"\nTraining data analysis:")
        print(f"Unique combined labels in training data: {len(labels)}")
        print("All labels:")
        for i, label in enumerate(sorted(labels)):
            print(f"  {i:2d}: {label}")

        return len(labels)
    except Exception as e:
        print(f"Error reading training data: {e}")
        return None


if __name__ == "__main__":
    print("=== Checking PEFT-trained model ===")
    model_labels = check_peft_model_labels(BASE_MODEL, PEFT_MODEL_PATH)

    print("\n=== Checking training data ===")
    training_labels = count_training_labels(TRAINING_DATA)

    print("\n=== Summary ===")
    if model_labels:
        print(f"Trained PEFT model expects: {model_labels} labels")
    if training_labels:
        print(f"Training data contains: {training_labels} unique labels")

    if model_labels and training_labels and model_labels != training_labels:
        print("⚠️  MISMATCH: Model and training data have different label counts!")
    elif model_labels and training_labels:
        print("✅ Model and training data label counts match")
