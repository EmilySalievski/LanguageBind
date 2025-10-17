import json

def load_json_file(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json_file(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def build_prompt_map(json2_data):
    # Map from original prompt → modified prompt
    return {entry["prompt"]: entry["modified_prompt"] for entry in json2_data}

def augment_json1_with_modified_prompts(json1_data, prompt_map):
    for video in json1_data.get("videos", []):
        for clip in video.get("clips", []):
            for annotation in clip.get("annotations", []):
                for level in ["original", "moderate", "difficult"]:
                    key = f"query_{level}"
                    rewritten_key = f"{key}_rewritten"
                    original_prompt = annotation.get(key)

                    if original_prompt in prompt_map:
                        annotation[rewritten_key] = prompt_map[original_prompt]
                    else:
                        annotation[rewritten_key] = None  # or raise/log if needed
                        print(f"No match found for: '{original_prompt}'")
    return json1_data

def main():
    # Paths to your files (change if needed)
    json1_path = 'intent_dataset/merged_datatset.json'
    json2_path = 'intent_dataset/rewritten_prompts_full_model.json'
    output_path = 'intent_dataset/merged_dataset_matched_prompts.json'

    # Load data
    json1_data = load_json_file(json1_path)
    json2_data = load_json_file(json2_path)

    # Build mapping and augment
    prompt_map = build_prompt_map(json2_data)
    updated_json1 = augment_json1_with_modified_prompts(json1_data, prompt_map)

    # Save updated JSON
    save_json_file(output_path, updated_json1)
    print(f"Updated JSON saved to '{output_path}'")

if __name__ == "__main__":
    main()
