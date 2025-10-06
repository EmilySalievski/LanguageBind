from datasets import Dataset, DatasetDict
import json

def load_intent_dataset_jsonl(path="data/intention_estimation_100.jsonl"):
    """
    Load JSONL dataset for intention estimation (fine-tune only on `task_type`).
    Returns a DatasetDict with train/test splits.
    """
    data = []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            data.append({
                "text": entry["prompt"],
                "label": entry["task_type"]  # ✅ only use task_type
            })

    dataset = Dataset.from_list(data)
    return dataset.train_test_split(test_size=0.2)

# from datasets import Dataset, DatasetDict
# import json
#
# def load_intent_dataset_jsonl(path="data/intention_estimation_100.jsonl"):
#     """
#     Load JSONL dataset for intention estimation.
#     Returns a DatasetDict with train/test splits.
#     """
#     data = []
#     with open(path, "r") as f:
#         for line in f:
#             entry = json.loads(line)
#             # Flatten labels into a single 'label' string per example for classification
#             # You can also store separate fields if doing multi-head training later
#             entry["label"] = "|".join([
#                 entry["task_type"],
#                 entry["output_modality"],
#                 entry["complexity"],
#                 entry["temporal_context"],
#                 entry["spatial_context"]
#             ])
#             data.append({"text": entry["prompt"], "label": entry["label"]})
#
#     dataset = Dataset.from_list(data)
#     return dataset.train_test_split(test_size=0.2)
