from datasets import Dataset
import pandas as pd

def load_intent_dataset(path="data/intents.csv"):
    """
    Load intents CSV and split into train/test.
    Returns a DatasetDict with 'train' and 'test' splits.
    """
    df = pd.read_csv(path, dtype=str)  # ensures all columns are strings
    df["label"] = df["label"].astype(str)
    dataset = Dataset.from_pandas(df, preserve_index=False)  # avoid adding 'index' column
    return dataset.train_test_split(test_size=0.2)
