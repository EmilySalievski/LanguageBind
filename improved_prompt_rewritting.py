from prompt_rewriting_module import prompt_rewriting
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_query(query):
    # Skip if it already exists
    if query.get("modified_prompt"):
        return query
    prompt = query["prompt"]
    task_type = query["task_type"]
    output_modality = query["output_modality"]
    complexity = query["complexity"]
    temporal_context = query["temporal_context"]
    spatial_context = query["spatial_context"]

    intent = {
        "task_type": task_type,
        "output_modality": output_modality,
        "complexity": complexity,
        "temporal_context": temporal_context,
        "spatial_context": spatial_context
    }

    module = prompt_rewriting.PromptRewritingModule(prompt, intent)
    modified_prompt = module.genAIsResponse()
    print(f"-----\nPrompt: {prompt}\nconverted to\nModified Prompt: {modified_prompt}\n-----")
    query["modified_prompt"] = modified_prompt

    return query


def batch_run_prompt_rewriting_module(input_file: str, output_file: str):
    with open(input_file, 'r') as f:
        query_data = json.load(f)

    modified_queries = []
    counter = 0

    # Adjust number of workers based on your CPU or use default
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_query, query) for query in query_data]

        for future in as_completed(futures):
            modified_queries.append(future.result())
            counter += 1
            # Save progress every 100 results
            if counter % 100 == 0:
                print(f"Processed {counter} queries. Saving intermediate results...")
                with open(output_file, 'w') as f:
                    json.dump(modified_queries, f, indent=2)

    with open(output_file, 'w') as f:
        json.dump(modified_queries, f, indent=2)

if __name__ == "__main__":
    batch_run_prompt_rewriting_module(
        "intent_dataset/ryan_intent_dataset.json", 
        "intent_dataset/ryan_intent_dataset.json"
    )