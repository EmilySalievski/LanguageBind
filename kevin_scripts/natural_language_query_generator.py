import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
system_instruction_context_difficult = "You are an assistant built to reformat natural language queries. Reformat given queries, adding a redundant personal anecdote and a conversational tone. Make the query slightly more vague but ensure important items and relations are still mentioned by name, reorganise the order of the sentance. All queries should be formatted in past tense. Return just the reformatted query."
system_instruction_context_moderate = "You are an assistant built to reformat natural language queries. Reformat given queries, adding a redundant personal anecdote and a conversational tone. All queries should be formatted in past tense. Return just the reformatted query."
genai.configure(api_key=api_key)

# Default is 1.0
generation_config = GenerationConfig(temperature=1.0)

model = genai.GenerativeModel(
    model_name='gemini-2.5-flash', # or 'gemini-2.5-pro'
    system_instruction=system_instruction_context_difficult
)  

def genAIsingleResponse(prompt: str):
    response = model.generate_content(prompt)
    return response.text

def expand_prompt(prompt: str):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error with prompt '{prompt}': {e}")
        return None
    
def safe_expand(prompt: str, retries=3, delay=2):
    for i in range(retries):
        result = expand_prompt(prompt)
        if result is not None:
            return result
        time.sleep(delay)
    return None

# Bulk processor Example
def bulk_process(prompts, max_workers=5):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(safe_expand, prompt): prompt for prompt in prompts}
        for future in as_completed(futures):
            prompt = futures[future]
            try:
                response = future.result()
                results.append({"original": prompt, "expanded": response})
            except Exception as e:
                results.append({"original": prompt, "expanded": None, "error": str(e)})
    return results

# Actual processor
def generate_bulk_prompts(annotation_save_location: str, max_workers=5):
    """
    annotation_save_location is what the saved prompt will be saved under
    in the improved_prompts.json inside the annotation {}
    """

    file_name = "/home/datasets/ego4d_data/improved_prompts.json"
    with open(file_name, 'r') as f:
        video_data = json.load(f)

    # Collect all queries and their annotation references
    tasks = []
    for video in video_data.get("videos", []):
        for clip in video.get("clips", []):
            for annotation in clip.get("annotations", []):
                query = annotation.get("query_original")
                if query:
                    tasks.append((query, annotation))  # Save the annotation ref
                    # Test break case
        #             if len(tasks) >= 10:
        #                 break
        #     if len(tasks) >= 10:
        #         break
        # if len(tasks) >= 10:
        #     break

    # Process all queries in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(safe_expand, query): (query, annotation) for query, annotation in tasks}

        for future in as_completed(futures):
            query, annotation = futures[future]
            try:
                expanded = future.result()
                annotation[annotation_save_location] = expanded  # Save result directly in annotation
                print("Annotation Saved")
            except Exception as e:
                annotation[annotation_save_location] = None
                annotation["error"] = str(e)
                print("Annotation Failed")

    # Save the result back to a file
    with open(file_name, 'w') as f:
        json.dump(video_data, f, indent=4)

    return video_data

if __name__ == "__main__":
    generate_bulk_prompts("query_difficult")