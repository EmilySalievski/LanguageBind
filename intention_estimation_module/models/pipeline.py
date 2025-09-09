import yaml
from models.classifier import DebertaClassifier
from models.fallback import MistralFallback

class IntentionPipeline:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.threshold = cfg["confidence_threshold"]
        self.classifier = DebertaClassifier(cfg["classifier_model"], "data/labels.json")
        self.fallback = MistralFallback(cfg["fallback_model"])

    def process(self, query: str):
        result = self.classifier.predict(query)
        if result["confidence"] >= self.threshold:
            return {"method": "classifier", **result}
        else:
            llm_result = self.fallback.infer_intent(query)
            return {"method": "fallback", "llm_output": llm_result}
