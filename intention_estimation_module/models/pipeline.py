import json
from models.classifier import IntentionClassifier
from models.fallback import MistralFallback


class IntentionPipeline:
    def __init__(self):
        self.classifier = IntentionClassifier(
            model_name="microsoft/deberta-v3-large",
            training_data_path="data/ground_truth.jsonl",
            peft_path="models/fine_tuned_classifier",
            confidence_threshold=0.4
        )

        # Initialize fallback LLM once
        self.fallback = MistralFallback()

    def process(self, query: str):
        """
        Main pipeline: classify query -> optionally use fallback -> return structured result
        """
        # Step 1: Predict task_type using DeBERTa classifier
        task_pred = self.classifier.predict_task_type(query)
        confidence = task_pred["confidence"]
        task_type = task_pred["task_type"]

        task_type = task_type.replace("_", " ").lower()

        print(f"[Classifier] task_type='{task_type}' (confidence={confidence:.3f})")

        # Step 2: Decide fallback usage
        if self.classifier.should_use_fallback(confidence):
            print("⚠️ Low confidence — using Mistral LLM for full classification.")
            mistral_result = self.fallback.infer_intent(query)
        else:
            print("✅ Confident task type — using Mistral for detailed intention refinement.")
            mistral_result = self.fallback.infer_remaining_intent(query, fixed_task_type=task_type)


        # Step 3: Merge results
        final_result = {
            "prompt": query,
            "task_type": task_type if not self.classifier.should_use_fallback(confidence) else mistral_result.get("task_type", task_type),
            "confidence": confidence,
            **{k: v for k, v in mistral_result.items() if k not in ["task_type", "confidence"]}
        }

        return json.dumps(final_result, indent=2)

# import yaml
# from models.classifier import DebertaClassifier
# from models.fallback import MistralFallback
#
#
# class IntentionPipeline:
#     def __init__(self, config_path="config/config.yaml"):
#         # Load configuration
#         with open(config_path) as f:
#             cfg = yaml.safe_load(f)
#
#         self.threshold = cfg.get("confidence_threshold", 0.6)
#
#         # Load classifier with combined labels extracted from training data
#         self.classifier = DebertaClassifier(
#             model_name=cfg["classifier_model"],
#             training_data_path="data/ground_truth.jsonl",
#             peft_path=cfg.get("peft_model"),
#             temps_path="data/temps.json"
#         )
#
#         # Fallback LLM
#         self.fallback = MistralFallback(cfg["fallback_model"])
#
#     def process(self, query: str):
#         result = self.classifier.predict(query)
#
#         # Print confidence to console
#         print(f"[Confidence] {result['confidence']:.4f}")
#
#         if result["confidence"] >= self.threshold:
#             # Split combined label back into fields
#             task_type, output_modality, complexity, temporal_context, spatial_context = result["pred_label"].split("|")
#
#             return {
#                 "prompt": query,
#                 "task_type": task_type,
#                 "output_modality": output_modality,
#                 "complexity": complexity,
#                 "temporal_context": temporal_context,
#                 "spatial_context": spatial_context,
#             }
#         else:
#             llm_result = self.fallback.infer_intent(query)
#             return {
#                 "prompt": query,
#                 "method": "fallback",
#                 "llm_output": llm_result,
#             }


# import yaml
# from models.classifier import DebertaClassifier
# from models.fallback import MistralFallback
#
# class IntentionPipeline:
#     def __init__(self, config_path="config/config.yaml"):
#         # with open(config_path) as f:
#         #     cfg = yaml.safe_load(f)
#         # self.threshold = cfg["confidence_threshold"]
#         # self.classifier = DebertaClassifier(cfg["classifier_model"], "data/labels.json", temps_path="data/temps.json")
#
#         # Load configuration
#         with open(config_path) as f:
#             cfg = yaml.safe_load(f)
#
#         self.threshold = cfg.get("confidence_threshold", 0.6)
#
#         # Load classifier with combined labels extracted from training data
#         self.classifier = DebertaClassifier(
#             model_name=cfg["classifier_model"],
#             training_data_path="data/ground_truth.jsonl",
#             peft_path=cfg.get("peft_model"),
#             temps_path="data/temps.json"
#         )
#         # self.classifier = DebertaClassifier(
#         #     model_name=cfg["classifier_model"],
#         #     training_data_path="data/ground_truth.jsonl",  # automatically extracts 29 labels
#         #     temps_path="data/temps.json",
#         #     peft_path=cfg.get("peft_model")  # optional, points to LoRA adapter
#         # )
#
#         # call fallback
#         self.fallback = MistralFallback(cfg["fallback_model"])
#
#     def process(self, query: str):
#         result = self.classifier.predict(query)
#         if result["confidence"] >= self.threshold:
#             return {"method": "classifier", **result}
#         else:
#             llm_result = self.fallback.infer_intent(query)
#             return {"method": "fallback", "llm_output": llm_result}
