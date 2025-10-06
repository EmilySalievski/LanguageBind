from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
from difflib import get_close_matches

ALLOWED_OPTIONS = {
    "task_type": ["temporal localisation", "object detection", "event recognition", "question answering"],
    "output_modality": ["text", "image", "video"],
    "complexity": ["temporal", "simple", "causal"],
    "temporal_context": ["none", "during", "before", "after"],
    "spatial_context": ["none", "inside", "in front", "left of", "right of", "behind", "on top of", "under"]
}


class MistralFallback:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # ------------------------------
    # Full fallback (classifier not confident)
    # ------------------------------
    def infer_intent(self, text: str) -> dict:
        prompt = self._build_guided_prompt(text)
        return self._run_llm_and_parse(prompt, text)

    # ------------------------------
    # Partial fallback (classifier confident)
    # ------------------------------
    def infer_remaining_intent(self, text: str, fixed_task_type: str) -> dict:
        prompt = self._build_guided_prompt(text, fixed_task_type=fixed_task_type)
        result = self._run_llm_and_parse(prompt, text)

        # Lock in classifier task_type
        result["task_type"] = fixed_task_type
        return {
            "output_modality": result.get("output_modality", "unknown"),
            "complexity": result.get("complexity", "unknown"),
            "temporal_context": result.get("temporal_context", "unknown"),
            "spatial_context": result.get("spatial_context", "unknown"),
        }

    # ------------------------------
    # Helper: Build deterministic prompt
    # ------------------------------
    def _build_guided_prompt(self, query: str, fixed_task_type: str = None) -> str:
        base_guidelines = """
        You are an intention extraction model. Output JSON strictly with:
        "prompt", "task_type", "output_modality", "complexity",
        "temporal_context", "spatial_context".

        Use ONLY these allowed options:
        - task_type: temporal localisation, object detection, event recognition, question answering
        - output_modality: text, image, video
        - complexity: temporal, simple, causal
        - temporal_context: none, during, before, after
        - spatial_context: none, inside, in front, left of, right of, behind, on top of, under

        Follow these explicit rules:
        - If the query includes "when", "before", "after", or "during", prefer task_type="temporal localisation".
        - If it asks to find, detect, identify, or locate something → task_type="object detection".
        - If it asks what happened or describes an event → task_type="event recognition".
        - If it is phrased as a question or asks for reasoning → task_type="question answering".
        - Use output_modality="text" unless the query explicitly mentions images or videos.
        - Use "temporal" complexity when reasoning about time; "causal" if explaining cause/effect; otherwise "simple".
        - Use "none" for temporal/spatial context unless explicitly implied (e.g., "before", "behind", etc.).
        """

        if fixed_task_type:
            base_guidelines += f"\nThe classifier has already determined task_type='{fixed_task_type}'. Do NOT change it.\n"

        prompt = f"{base_guidelines}\nQuery: {query}\nRespond with valid JSON only."
        return prompt

    # ------------------------------
    # Helper: Determine output_modality
    # ------------------------------
    def _determine_output_modality(self, task_type, query):
        # Query mentions image → image
        if re.search(r"\b(image|picture|photo)\b", query, re.I):
            return "image"
        # Query mentions video or task implies temporal segment → video
        elif re.search(r"\b(video|clip|footage)\b", query, re.I) or task_type in ["temporal localisation",
                                                                                  "event recognition"]:
            return "video"
        # Object detection → image
        elif task_type == "object detection":
            return "image"
        return "unknown"

    # ------------------------------
    # Helper: Run model + robust JSON parsing
    # ------------------------------
    def _run_llm_and_parse(self, prompt: str, text: str) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9
        )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        parsed = {
            "prompt": text,
            "task_type": "unknown",
            "output_modality": "unknown",
            "secondary_output_modality": "text",
            "complexity": "unknown",
            "temporal_context": "none",
            "spatial_context": "none"
        }

        def _extract_json(text):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return match.group()
            return "{}"

        try:
            candidate = json.loads(_extract_json(result))
            parsed["prompt"] = candidate.get("prompt", text)
            for field in ["task_type", "complexity"]:
                val = candidate.get(field, "unknown")
                if val not in ALLOWED_OPTIONS.get(field, []):
                    matches = get_close_matches(val, ALLOWED_OPTIONS.get(field, []), n=1, cutoff=0.6)
                    parsed[field] = matches[0] if matches else "unknown"
                else:
                    parsed[field] = val

            # temporal_context and spatial_context default to 'none'
            parsed["temporal_context"] = candidate.get("temporal_context", "none")
            if parsed["temporal_context"] not in ALLOWED_OPTIONS["temporal_context"]:
                parsed["temporal_context"] = "none"

            parsed["spatial_context"] = candidate.get("spatial_context", "none")
            if parsed["spatial_context"] not in ALLOWED_OPTIONS["spatial_context"]:
                parsed["spatial_context"] = "none"

            # Determine output_modality explicitly
            parsed["output_modality"] = self._determine_output_modality(parsed["task_type"], text)
        except json.JSONDecodeError:
            pass

        return parsed

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import json
# from difflib import get_close_matches
#
# ALLOWED_OPTIONS = {
#     "task_type": ["temporal localisation", "object detection", "event recognition", "question answering"],
#     "output_modality": ["text", "image", "video"],
#     "complexity": ["temporal", "simple", "causal"],
#     "temporal_context": ["none", "during", "before", "after"],
#     "spatial_context": ["none", "inside", "in front", "left of", "right of", "behind", "on top of", "under"]
# }
#
# class MistralFallback:
#     def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16,
#             device_map="auto"
#         )
#
#     def infer_intent(self, text: str) -> dict:
#         """
#         Returns JSON strictly in the ground_truth format, selecting
#         values only from the allowed options.
#         """
#         prompt = f"""
#         You are an intention extraction system.
#         Analyze the query and output a JSON with keys:
#         "prompt", "task_type", "output_modality", "complexity",
#         "temporal_context", "spatial_context".
#
#         Only choose values from the allowed options:
#         task_type: {ALLOWED_OPTIONS['task_type']}
#         output_modality: {ALLOWED_OPTIONS['output_modality']}
#         complexity: {ALLOWED_OPTIONS['complexity']}
#         temporal_context: {ALLOWED_OPTIONS['temporal_context']}
#         spatial_context: {ALLOWED_OPTIONS['spatial_context']}
#
#         Query: {text}
#         Respond with valid JSON only.
#         """
#
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=256,
#             temperature=0.2,
#             top_p=0.95
#         )
#
#         result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#         # Default fallback
#         parsed = {
#             "prompt": text,
#             "task_type": "unknown",
#             "output_modality": "unknown",
#             "complexity": "unknown",
#             "temporal_context": "unknown",
#             "spatial_context": "unknown"
#         }
#
#         # Attempt to parse JSON
#         try:
#             candidate = json.loads(result)
#             parsed["prompt"] = candidate.get("prompt", text)
#
#             # Map each field to the closest allowed option
#             for field in ["task_type", "output_modality", "complexity", "temporal_context", "spatial_context"]:
#                 val = candidate.get(field, "unknown")
#                 if val not in ALLOWED_OPTIONS[field]:
#                     # pick closest match or fallback to 'unknown'
#                     matches = get_close_matches(val, ALLOWED_OPTIONS[field], n=1, cutoff=0.6)
#                     parsed[field] = matches[0] if matches else "unknown"
#                 else:
#                     parsed[field] = val
#         except json.JSONDecodeError:
#             pass  # fallback defaults remain
#
#         return parsed
