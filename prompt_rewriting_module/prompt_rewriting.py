import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
system_instruction_general = """
You are a prompt rewriting assistant. Your goal is to take a user's input prompt
and user intent to rewrite it into a clearer, more specific, and detailed form.
Remove information from the prompt that is irrelevant.

Guidelines:
- Directly identify items by their names (if known) or descriptors instead of using pronouns
  (e.g., “the red cup on the table” instead of “it”).
- Avoid oversimplifying; minimal prompts often reduce performance. Ensure the rewritten prompt
  is sufficiently descriptive.
- Rewrite the prompt into a clear, concise, and directed query. Prefer formulations like
  “Location of object X” instead of vague or open-ended phrasing.
- Exclude irrelevant or noisy context unless it contributes meaningfully to the query.
- Be aware that the model may struggle with tasks that require reading text inside a video
  or identifying items by brand names. Rewritten prompts should replace such references with the 
  generic object description provided in the user's prompt (e.g., rewrite “Russell Stover bag” as “chocolate bag”).

Your output should always be a single improved prompt in natural language that follows these guidelines.
"""

genai.configure(api_key=api_key)

class PromptRewritingModule():
    
    def __init__(self, original_prompt, user_intent): 
        self.original_prompt = original_prompt
        self.user_intent = user_intent

        if isinstance(user_intent, str): # assuming user_intent is passed through as json originally
            self.user_intent = json.loads(user_intent)
        else:
            self.user_intent = user_intent 

        self.task_type = self.user_intent.get("task_type") # Temporal Localisation, Object Detection, Event Recognition, Question Answering
        self.output_modality = self.user_intent.get("output_modality") # Text, Image, Video - not that relevant for our specific usage of languagebind, but can be helpful for future extensions
        self.complexity = self.user_intent.get("complexity") #  Temporal, Simple, Causal
        self.temporal_context = self.user_intent.get("temporal_context") # None, During, Before, After
        self.spatial_context = self.user_intent.get("spatial_context") # None, Inside, In front, Left of, Right of, Behind, On top of, Under

        self.task_extra_information = None
        self.modality_extra_information = None
        self.complexity_extra_information = None
        self.temporal_extra_information = None
        self.spatial_extra_information = None

        # Default is 1.0
        # self.generation_config = GenerationConfig(temperature=1.0)

        self.model = genai.GenerativeModel(
            model_name='gemini-2.5-flash', # or 'gemini-2.5-pro'
            system_instruction=system_instruction_general
        )  

    def task_info(self): # giving few-shot example prompts as extra context
        if str(self.task_type).upper() == "TEMPORAL LOCALISATION":
            self.task_extra_information = """
            This is a temporal localisation task. Prompts can be rewritten like so:
            Prompt: "Can you just tell whenever that cup falls off the table? I remember I once dropped my mug and it shattered everywhere - it was a huge pain to clean up! That was ages ago though.

            Rewritten prompt: "Locate when the cup falls off the table."
            """
        if str(self.task_type).upper() == "OBJECT DETECTION":
            self.task_extra_information = """
            This is a temporal localisation task. Prompts can be rewritten like so:
            Prompt: "..."

            Rewritten "..."
            """
        if str(self.task_type).upper() == "EVENT RECOGNITION":
            self.task_extra_information = """
            This is a temporal localisation task. Prompts can be rewritten like so:
            Prompt: "..."

            Rewritten "..."
            """
        if str(self.task_type).upper() == "QUESTION ANSWERING":
            self.task_extra_information = """
            This is a temporal localisation task. Prompts can be rewritten like so:
            Prompt: "..."

            Rewritten "..."
            """

    def modality_info(self):
        pass

    def complexity_info(self):
        pass

    def temporal_info(self):
        pass

    def spatial_info(self):
        pass
            
    def genAIsResponse(self):
        # Collect extra contexts
        self.task_info()
        self.modality_info()
        self.complexity_info()
        self.temporal_info()
        self.spatial_info()

        # Combine all extra context info into one string
        extra_info = ""
        if self.task_extra_information:
            extra_info += self.task_extra_information + "\n"
        if self.modality_extra_information:
            extra_info += self.modality_extra_information + "\n"
        if self.complexity_extra_information:
            extra_info += self.complexity_extra_information + "\n"
        if self.temporal_extra_information:
            extra_info += self.temporal_extra_information + "\n"
        if self.spatial_extra_information:
            extra_info += self.spatial_extra_information + "\n"

        # Send original prompt + all extra info together as "user" input
        response = self.model.generate_content([
            {
                "role": "user",
                "parts": [
                    f"Original prompt:\n{self.original_prompt.strip()}\n\nExtra context:\n{extra_info.strip()}"
                ]
            }
        ])
        return response.text

if __name__ == "__main__":
    intent = {
        "task_type": "Temporal Localisation",
        "output_modality": "Video",
        "complexity": "Simple",
        "temporal_context": "During",
        "spatial_context": "None"
    }
    prompt = """
    In this birthday party video, there are kids running around, music playing, and people laughing. 
    It made me think of when I was younger and I hated when balloons popped because it was so loud. 
    Anyway, I just want to know when the child actually pops the balloon in this video.
    """
    module = PromptRewritingModule(prompt, intent)
    print(module.genAIsResponse())