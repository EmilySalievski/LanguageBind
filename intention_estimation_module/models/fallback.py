from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json


class MistralFallback:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def infer_intent(self, text: str) -> dict:
        """
        Consistent with the old API.
        Returns JSON with 'intent' and 'explanation'.
        """
        prompt = f"""
        You are an intention extraction system.
        Analyze the query and output a JSON with 'intent' and 'explanation'.
        Query: {text}
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.95
        )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Try to parse JSON safely
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            parsed = {"intent": "unknown", "explanation": result.strip()}

        return parsed
