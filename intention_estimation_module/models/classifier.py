import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

class DebertaClassifier:
    def __init__(self, model_name, labels_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, use_safetensors=True
        )
        with open(labels_path) as f:
            self.labels = json.load(f)["intents"]

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
            confidence, idx = torch.max(probs, dim=0)

        return {
            "intent": self.labels[idx],
            "confidence": confidence.item()
        }

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
            confidence, idx = torch.max(probs, dim=0)

        return {
            "intent": self.labels[idx],
            "confidence": confidence.item()
        }
