import json

def getNLQs(nlq_json_file: str, existing_dict: dict|None = None):
    # Returns a dictionary of videos to be exported
    # open and read the json annotations file
    with open(nlq_json_file, "r") as f:
        annotation_data = json.load(f)

    # Load the list of video UIDs from the .txt file
    with open("ego4d_subset_ids.txt", "r") as f:
        video_uids = [line.strip() for line in f if line.strip()]

    # This will be the final dictionary to be exported
    if existing_dict is None:
        output_data = {"videos": []}
    else:
        output_data = existing_dict

    for video in annotation_data.get("videos", []):
        video_uid = video.get("video_uid")

        if video_uid in video_uids:
            # Create a new video entry for the output
            new_video_entry = {
                "video_uid": video_uid,
                "clips": []
            }
            
            # This dictionary will help group annotations by clip_uid
            clips_by_uid = {}

            # Loop over clips → annotations
            for clip in video.get("clips", []):
                clip_uid = clip.get("clip_uid")
                video_start_sec = clip.get("video_start_sec")
                video_end_sec = clip.get("video_end_sec")

                # If we haven't seen this clip_uid yet, create its entry
                if clip_uid not in clips_by_uid:
                    clips_by_uid[clip_uid] = {
                        "clip_uid": clip_uid,
                        "clip_start_sec": video_start_sec,
                        "clip_end_sec": video_end_sec,
                        "annotations": []
                    }

                # natural language query data
                for annotation in clip.get("annotations", []):
                    for lang_query in annotation.get("language_queries", []):
                        query = lang_query.get("query")
                        query_start_sec = lang_query.get("video_start_sec")
                        query_end_sec = lang_query.get("video_end_sec")
                        
                        if query:
                            # Append the annotation to the correct clip
                            clips_by_uid[clip_uid]["annotations"].append({
                                "query_original": query,
                                "query_moderate": None,  # Assuming these fields are not available
                                "query_difficult": None,
                                "query_start": query_start_sec,
                                "query_end": query_end_sec
                            })
            
            # Add all the processed clips to the new_video_entry
            new_video_entry["clips"] = list(clips_by_uid.values())
            output_data["videos"].append(new_video_entry)
    return output_data

def jsonToFile(jsonDict: dict, file: str):
    json_output_string = json.dumps(jsonDict, indent=4)
    if json_output_string:
        with open(file, "w") as out_file:
            out_file.write(json_output_string)
            print("Successfully wrote data to " + file)

if __name__ == "__main__":
    video_dict = {"videos": []}
    getNLQs("/home/datasets/ego4d_data/v2/annotations/nlq_train.json", video_dict)
    print("nlq_train: " + str(len(video_dict["videos"])))
    getNLQs("/home/datasets/ego4d_data/v2/annotations/nlq_val.json", video_dict)
    print("nlq_val: " + str(len(video_dict["videos"])))
    # 503 Video annotations found - Where are the rest?
    jsonToFile(video_dict, "improved_prompts.json")

