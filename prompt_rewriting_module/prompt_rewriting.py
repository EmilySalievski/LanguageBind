import os
import google.generativeai as genai
from dotenv import load_dotenv
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
            Prompt: "I once dropped my mug and it shattered everywhere. It was such a pain to clean up, and I still remember 
                     it even though it was ages ago. Could you let me know if that cup falls off the table? I'd rather not go 
                     through that hassle again.”

            Rewritten prompt: "Locate when the cup falls off the table."
            """
        if str(self.task_type).upper() == "OBJECT DETECTION":
            self.task_extra_information = """
            This is an object detection task. Prompts can be rewritten like so:
            Prompt: "I'm going for a picnic tomorrow and I want to make sure the basket is packed. Could you tell me if there's 
                     a Mount Franklin bottle in the scene so I don't forget?"

            Rewritten prompt: "Detect the water bottle."
            """
        if str(self.task_type).upper() == "EVENT RECOGNITION":
            self.task_extra_information = """
            This is an event recognition task. Prompts can be rewritten like so:
            Prompt: "I love birthday parties. Could you check if the person in the video is blowing out the candles on the cake?"

            Rewritten prompt: "Show when the candles are blown out."
            """
        if str(self.task_type).upper() == "QUESTION ANSWERING":
            self.task_extra_information = """
            This is a question answering task. Prompts can be rewritten like so:
            Prompt: "It's my son's fifth birthday today and I'm still setting up! How many balloons are in the video? I need 
                     to check before the party."

            Rewritten prompt: "Location the balloons in the video." 

            Prompt: "My favourite colour is red, and I always notice it everywhere. What colour is his sweater in the video?"
            
            Rewritten prompt: "Locate the man wearing a sweater."
            """

    def modality_info(self):
        pass

    def complexity_info(self):
        if str(self.complexity).upper() == "SIMPLE":
            self.complexity_extra_information = """
            This is a simple complexity task. Prompts can be rewritten like so:
            Prompt: "I have two cats at home, one of them is black and always hides under the couch. By the way, can you see 
                     if there is a cat in the video?"
            
            Rewritten prompt: "Detect the cat."
            """

        if str(self.complexity).upper() == "CAUSAL":
            self.complexity_extra_information = """
            This is a causal complexity task. Prompts can be rewritten like so:
            Prompt: "Last week I dropped a glass on my kitchen floor, and it shattered everywhere. In the video, the vase
                     broke because something knocked it over. Could you tell me what caused the vase to break?"
            
            Rewritten prompt: "Show when the vase is knocked over."
            """

        if str(self.complexity).upper() == "TEMPORAL":
            self.complexity_extra_information = """
            This is a simple complexity task. Prompts can be rewritten like so:
            Prompt: "I remember when I used to play soccer as a kid, I always kicked the ball too early. In the video, the boy 
                     first grabs the ball, then later he kicks it. Can you find when he kicks the ball?"
            
            Rewritten prompt: "Detect when the boy kicks the ball."
            """

    def temporal_info(self):
        if str(self.temporal_context).upper() == "DURING":
            self.temporal_extra_information = """
            This is a temporal context (During) task. Prompts can be rewritten like so:
            Prompt: "I always cheer loudest during the last lap of a race. In the video, can you show the moment during the race when the runner crosses the finish line?"
            
            Rewritten prompt: "During the race, show the runner crossing the finish line."
            """

        if str(self.temporal_context).upper() == "BEFORE":
            self.temporal_extra_information = """
            This is a temporal context (Before) task. Prompts can be rewritten like so:
            Prompt: "Before every birthday, my family sets up decorations. In the video, can you find what happens before the child opens the present?"
            
            Rewritten prompt: "Before the child opens the present."
            """

        if str(self.temporal_context).upper() == "AFTER":
            self.temporal_extra_information = """
            This is a temporal context (After) task. Prompts can be rewritten like so:
            Prompt: "After I finish my coffee, I usually check my phone. In the video, can you show what happens after the man drops his phone?"
            
            Rewritten prompt: "After the man drops his phone."
            """

    def spatial_info(self):
        if str(self.spatial_context).upper() == "INSIDE":
            self.spatial_extra_information = """
            This is a spatial context (Inside) task. Prompts can be rewritten like so:
            Prompt: "I once hid inside a car trunk as a prank. In the video, can you show the dog inside the car?"
            
            Rewritten prompt: "Dog inside the car."
            """

        if str(self.spatial_context).upper() == "IN FRONT":
            self.spatial_extra_information = """
            This is a spatial context (In front) task. Prompts can be rewritten like so:
            Prompt: "At school, I hated standing in front of everyone during speeches. In the video, can you show the person standing in front of the building?"
            
            Rewritten prompt: "Person in front of the building."
            """

        if str(self.spatial_context).upper() == "LEFT OF":
            self.spatial_extra_information = """
            This is a spatial context (Left of) task. Prompts can be rewritten like so:
            Prompt: "I always keep my notebook to the right of my laptop. In the video, can you show the chair to the left of the table?"
            
            Rewritten prompt: "Chair left of the table."
            """

        if str(self.spatial_context).upper() == "RIGHT OF":
            self.spatial_extra_information = """
            This is a spatial context (Right of) task. Prompts can be rewritten like so:
            Prompt: "I like to keep my living room well-light with lots of lights. Can you show the lamp to the right of the sofa?"
            
            Rewritten prompt: "Lamp right of the sofa."
            """

        if str(self.spatial_context).upper() == "BEHIND":
            self.spatial_extra_information = """
            This is a spatial context (Behind) task. Prompts can be rewritten like so:
            Prompt: "I took my dog to the dog park earlier today and took this video. In the video, can you show the dog behind the fence? "
            
            Rewritten prompt: "Dog behind the fence."
            """

        if str(self.spatial_context).upper() == "ON TOP OF":
            self.spatial_extra_information = """
            This is a spatial context (On top of) task. Prompts can be rewritten like so:
            Prompt: "I love reading - it's one of my favourite pastimes. I had to pack all my books away when I was moving out.
                     I think I misplaced my books, can you show the box on top of the table?"
            
            Rewritten prompt: "Box on top of the table."
            """

        if str(self.spatial_context).upper() == "UNDER":
            self.spatial_extra_information = """
            This is a spatial context (Under) task. Prompts can be rewritten like so:
            Prompt: "As a child I used to crawl under the bed during storms. In the video, can you show the cat under the chair?"
            
            Rewritten prompt: "Cat under the chair."
            """

            
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