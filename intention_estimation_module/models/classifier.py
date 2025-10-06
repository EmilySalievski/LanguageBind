import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import json


class IntentionClassifier:
    def __init__(self, model_name, training_data_path, peft_path=None, temps_path=None, confidence_threshold=0.4):
        """
        Load tokenizer, PEFT-adapted model, and prepare task type label mappings.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.confidence_threshold = confidence_threshold

        # Extract only task_type labels (not combined)
        self.task_labels = self._extract_unique_field(training_data_path, "task_type")
        self.id2label = {i: l for i, l in enumerate(self.task_labels)}
        self.label2id = {l: i for i, l in enumerate(self.task_labels)}

        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.task_labels),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=True,
        )

        if peft_path:
            self.model = PeftModel.from_pretrained(base_model, peft_path)
        else:
            self.model = base_model

        # Load temperature calibration if available
        if temps_path:
            with open(temps_path) as f:
                self.temperatures = json.load(f)
        else:
            self.temperatures = {label: 1.0 for label in self.task_labels}

        self.device = next(self.model.parameters()).device

    def _extract_unique_field(self, training_file, field):
        """Extract sorted unique values for a single field from training data."""
        values = set()
        with open(training_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                values.add(entry[field])
        return sorted(values)

    def predict_task_type(self, text):
        """Predict only the task_type with confidence."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(0)

        T = 1.0
        probs = torch.nn.functional.softmax(logits / T, dim=-1)
        conf, pred_idx = torch.max(probs, dim=-1)
        pred_label = self.task_labels[pred_idx]

        return {
            "task_type": pred_label,
            "confidence": conf.item(),
            "probabilities": {self.task_labels[i]: p.item() for i, p in enumerate(probs)},
        }

    def should_use_fallback(self, confidence):
        """Decide whether to fall back to Mistral."""
        return confidence < self.confidence_threshold

# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from peft import PeftModel
# import json
#
#
# class DebertaClassifier:
#     def __init__(self, model_name, training_data_path, peft_path=None, temps_path=None):
#         """
#         Load tokenizer, PEFT-adapted model, and compute label mapping from training data
#         """
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#         # Extract combined labels from training data
#         self.labels = self._extract_combined_labels(training_data_path)
#         self.total_labels = len(self.labels)
#         self.id2label = {i: l for i, l in enumerate(self.labels)}
#         self.label2id = {l: i for i, l in enumerate(self.labels)}
#
#         # Load base model with correct number of labels
#         base_model = AutoModelForSequenceClassification.from_pretrained(
#             model_name,
#             num_labels=self.total_labels,
#             torch_dtype=torch.float16,
#             device_map="auto",
#             trust_remote_code=True,
#             use_safetensors=True,
#         )
#
#         # Load PEFT adapter if provided
#         if peft_path:
#             self.model = PeftModel.from_pretrained(base_model, peft_path)
#         else:
#             self.model = base_model
#
#         # Load calibrated temperatures or default to 1.0
#         if temps_path:
#             with open(temps_path) as f:
#                 self.temperatures = json.load(f)
#         else:
#             self.temperatures = {label: 1.0 for label in self.labels}
#
#         self.device = next(self.model.parameters()).device
#
#     def _extract_combined_labels(self, training_file):
#         """Return sorted list of unique combined labels in training data"""
#         labels = set()
#         with open(training_file, "r") as f:
#             for line in f:
#                 entry = json.loads(line)
#                 combined = "|".join([
#                     entry["task_type"],
#                     entry["output_modality"],
#                     entry["complexity"],
#                     entry["temporal_context"],
#                     entry["spatial_context"]
#                 ])
#                 labels.add(combined)
#         return sorted(labels)
#
#     def predict(self, text):
#         """Return predicted combined label and confidence"""
#         inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#
#         with torch.no_grad():
#             logits = self.model(**inputs).logits.squeeze(0)  # shape: [total_labels]
#
#         # Apply temperature scaling (optional)
#         T = 1.0  # or use self.temperatures if you want label-wise scaling
#         probs = torch.nn.functional.softmax(logits / T, dim=-1)
#         conf, pred_idx = torch.max(probs, dim=-1)
#         pred_label = self.labels[pred_idx]
#
#         return {
#             "pred_label": pred_label,
#             "confidence": conf.item(),
#             "probabilities": {self.labels[i]: p.item() for i, p in enumerate(probs)}
#         }



# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import json
# from peft import PeftModel
#
# class DebertaClassifier:
#     def __init__(self, model_name, labels_path, temps_path=None, peft_path=None):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#         # Determine number of labels from labels JSON
#         with open(labels_path) as f:
#             self.labels = json.load(f)
#         total_labels = sum(len(labels) for labels in self.labels.values())
#
#         # Load base model with correct number of labels
#         base_model = AutoModelForSequenceClassification.from_pretrained(
#             model_name,
#             num_labels=total_labels,  # <-- critical
#             torch_dtype=torch.float16,
#             device_map="auto",
#             trust_remote_code=True,
#             use_safetensors=True,
#         )
#
#         # Load PEFT adapter if given
#         if peft_path:
#             self.model = PeftModel.from_pretrained(base_model, peft_path)
#         else:
#             self.model = base_model
#
#         # Load calibrated temperatures or use 1.0 by default
#         if temps_path:
#             with open(temps_path) as f:
#                 self.temperatures = json.load(f)
#         else:
#             self.temperatures = {field: 1.0 for field in self.labels.keys()}
#
#         # Move model to device
#         self.device = next(self.model.parameters()).device
#
#         # Precompute index mapping per field
#         self.id_maps = {field: {label: i for i, label in enumerate(labels)}
#                         for field, labels in self.labels.items()}
#
#         # Compute total number of labels (assumes model outputs concatenated logits)
#         self.field_sizes = {field: len(labels) for field, labels in self.labels.items()}
#         self.total_labels = sum(self.field_sizes.values())
#
#     def predict(self, text):
#         """
#         Returns:
#             probs_per_field: dict with per-field predictions, probabilities, and confidences
#             overall_conf: mean confidence across all fields
#         """
#         inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             all_logits = outputs.logits.squeeze(0)  # [total_labels]
#
#         # Split logits per field
#         field_logits = {}
#         offset = 0
#         for field, size in self.field_sizes.items():
#             field_logits[field] = all_logits[offset: offset + size]
#             offset += size
#
#         # Apply temperature scaling and compute probabilities & confidences
#         probs_per_field = {}
#         confidences = []
#         for field, logits in field_logits.items():
#             T = self.temperatures.get(field, 1.0)
#             probs = torch.nn.functional.softmax(logits / T, dim=-1)
#             conf, pred_idx = torch.max(probs, dim=-1)
#             probs_per_field[field] = {
#                 "pred_label": self.labels[field][pred_idx],
#                 "confidence": conf.item(),
#                 "probabilities": {self.labels[field][i]: p.item() for i, p in enumerate(probs)}
#             }
#             confidences.append(conf)
#
#         # Overall confidence for fallback
#         overall_conf = torch.mean(torch.stack(confidences)).item()
#
#         return probs_per_field, overall_conf





# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import json
#
# class DebertaClassifier:
#     def __init__(self, model_name, labels_path):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         # self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, use_safetensors=True
#         )
#         with open(labels_path) as f:
#             self.labels = json.load(f)["intents"]
#
#     def predict(self, text):
#         inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#
#         device = next(self.model.parameters()).device
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
#             confidence, idx = torch.max(probs, dim=0)
#
#         return {
#             "intent": self.labels[idx],
#             "confidence": confidence.item()
#         }
#
#         # Tokenize
#         inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#
#         # Move inputs to the same device as the model
#         device = next(self.model.parameters()).device
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
#             confidence, idx = torch.max(probs, dim=0)
#
#         return {
#             "intent": self.labels[idx],
#             "confidence": confidence.item()
#         }
