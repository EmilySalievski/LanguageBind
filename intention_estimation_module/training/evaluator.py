from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_PATH = "models/fine_tuned_classifier"

# Load once
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def predict(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
        confidence, idx = torch.max(probs, dim=0)

    label_str = model.config.id2label[idx.item()]
    # Split the concatenated label back into fields if needed
    fields = label_str.split("|")
    return {
        "task_type": fields[0],
        "output_modality": fields[1],
        "complexity": fields[2],
        "temporal_context": fields[3],
        "spatial_context": fields[4],
        "confidence": confidence.item()
    }


if __name__ == "__main__":
    query = "Show me where the speaker compares PLA vs ABS"
    result = predict(query)
    print(result)


# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch
#
# MODEL_PATH = "models/fine_tuned_classifier"
#
#
# def predict(query):
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#     model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
#
#     # Retrieve label mapping
#     id2label = model.config.id2label
#
#     inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
#         confidence, idx = torch.max(probs, dim=0)
#
#     label_str = id2label[idx.item()]
#     return label_str, confidence.item()
#
#
# if __name__ == "__main__":
#     query = "Show me where the speaker compares PLA vs ABS"
#     label, conf = predict(query)
#     print(f"Predicted label: {label}, confidence: {conf:.2f}")