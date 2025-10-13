import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

def saveSoftmaxGraph(file: str, softmax_key: str):
    # Load data
    with open(file, 'r') as file:
        data = json.load(file)

    # Extract all softmax values
    softmax_values = [
        ann.get(softmax_key)
        for video in data["videos"]
        for clip in video["clips"]
        for ann in clip["annotations"]
        if ann.get(softmax_key) is not None
    ]

    # Convert to NumPy array
    softmax_array = np.array(softmax_values)

    # Compute statistics
    average = np.mean(softmax_array)
    std_dev = np.std(softmax_array)
    softmax_min = np.min(softmax_array)
    softmax_max = np.max(softmax_array)
    median = np.median(softmax_array)
    q1 = np.percentile(softmax_array, 25)
    q3 = np.percentile(softmax_array, 75)
    iqr = q3 - q1

    # Print results
    print(f"Count: {len(softmax_array)}")
    print(f"Mean: {average:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Min: {softmax_min:.4f}")
    print(f"Max: {softmax_max:.4f}")
    print(f"Median: {median:.4f}")
    print(f"Q1 (25th percentile): {q1:.4f}")
    print(f"Q3 (75th percentile): {q3:.4f}")
    print(f"IQR (Q3 - Q1): {iqr:.4f}")

    # Graph Data
    sns.histplot(softmax_array, bins=50, kde=True)
    plt.title(f"Distribution of {softmax_key}")
    plt.xlabel("Softmax Score")
    plt.ylabel("Count")

    plt.savefig(f"/home/cod022/stuff/{softmax_key}_plot.png")
    print(f"Plot saved as {softmax_key}_plot.png")
    return

def saveInverseTimeSoftmaxGraph(file: str, softmax_key: str):
    # Load data
    with open(file, 'r') as file:
        data = json.load(file)

    # Extract adjusted softmax values
    adjusted_softmax_values = []
    skipped = 0

    for video in data.get("videos", []):
        for clip in video.get("clips", []):
            for ann in clip.get("annotations", []):
                softmax_value = ann.get(softmax_key)
                query_length = ann.get("query_length_percentage")

                # Only process if both values are valid and query_length is not zero
                if softmax_value is not None and query_length and query_length > 0:
                    adjusted_score = softmax_value * (1 - query_length)
                    adjusted_softmax_values.append(adjusted_score)
                else:
                    skipped += 1

    if not adjusted_softmax_values:
        print("No valid adjusted softmax values found.")
        return

    print(f"Processed {len(adjusted_softmax_values)} values. Skipped {skipped} annotations.")

    # Convert to NumPy array
    adjusted_array = np.array(adjusted_softmax_values)

    # Compute Statistics
    average = np.mean(adjusted_array)
    std_dev = np.std(adjusted_array)
    softmax_min = np.min(adjusted_array)
    softmax_max = np.max(adjusted_array)
    median = np.median(adjusted_array)
    q1 = np.percentile(adjusted_array, 25)
    q3 = np.percentile(adjusted_array, 75)
    iqr = q3 - q1

    # Print results
    print(f"Count: {len(adjusted_array)}")
    print(f"Mean: {average:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Min: {softmax_min:.4f}")
    print(f"Max: {softmax_max:.4f}")
    print(f"Median: {median:.4f}")
    print(f"Q1 (25th percentile): {q1:.4f}")
    print(f"Q3 (75th percentile): {q3:.4f}")
    print(f"IQR (Q3 - Q1): {iqr:.4f}")

    # Graph Data
    sns.histplot(adjusted_array, bins=50, kde=True)
    plt.title(f"Adjusted Softmax (× 1 - query_length_percentage) for {softmax_key}")
    plt.xlabel("Adjusted Softmax Score")
    plt.ylabel("Count")

    plt.savefig(f"/home/cod022/stuff/{softmax_key}_adjusted_plot.png")
    print(f"Plot saved as {softmax_key}_adjusted_plot.png")
    return

def compare_p_values(file: str, value_key_1: str, value_key_2: str):
    # Load data
    with open(file, 'r') as f:
        data = json.load(f)

    # Extract values for both keys, ensuring they are aligned by position
    values_1 = []
    values_2 = []

    for video in data.get("videos", []):
        for clip in video.get("clips", []):
            for ann in clip.get("annotations", []):
                val1 = ann.get(value_key_1)
                val2 = ann.get(value_key_2)
                if val1 is not None and val2 is not None:
                    values_1.append(val1)
                    values_2.append(val2)

    # Check for equal lengths
    if len(values_1) != len(values_2):
        print("Error: Value lists are not the same length. Possible mismatch in paired data.")
        return

    # Optional: check for minimum number of pairs
    if len(values_1) < 10:
        print(f"Warning: Only {len(values_1)} valid pairs. Results may not be reliable.")

    # Run the Wilcoxon signed-rank test
    stat, p_value = wilcoxon(values_1, values_2)

    print(f"Wilcoxon test statistic: {stat}")
    print(f"P-value: {p_value}")

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print("Result: Statistically significant difference (reject H0)")
    else:
        print("Result: No statistically significant difference (fail to reject H0)")

if __name__ == "__main__":
    saveInverseTimeSoftmaxGraph("/home/datasets/ego4d_data/improved_prompts_with_softmax.json", "query_original_prompt_rewritting_ablation")