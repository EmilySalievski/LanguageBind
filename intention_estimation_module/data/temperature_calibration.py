import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import json

MODEL_NAME = "microsoft/deberta-v3-base"
VAL_FILE = "intention_estimation_100.jsonl"  # JSONL file

# Load label spaces from JSON
with open("labels.json") as f:
    label_spaces = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=sum(len(v) for v in label_spaces.values()),
    torch_dtype=torch.float16, device_map="auto", use_safetensors=True
)

# Load data
samples = [json.loads(l) for l in open(VAL_FILE)]
texts = [s["prompt"] for s in samples]

# Build target tensors
targets = {}
offset = 0
for field, labels in label_spaces.items():
    idmap = {l: i for i, l in enumerate(labels)}
    targets[field] = torch.tensor([idmap[s[field]] for s in samples])

# Get logits (assuming you trained the model to output concatenated logits for all fields)
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(next(model.parameters()).device)
with torch.no_grad():
    all_logits = model(**inputs).logits

# Split logits per field
field_logits = {}
offset = 0
for field, labels in label_spaces.items():
    n = len(labels)
    field_logits[field] = all_logits[:, offset:offset+n]
    offset += n

device = next(model.parameters()).device

# === Temperature scaling per field ===
def calibrate(logits, y_true):
    T = torch.nn.Parameter(torch.ones(1, device=device))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, y_true.to(device))
        loss.backward()
        return loss

    opt.step(closure)
    return T.item()

def compute_ece(logits, labels, n_bins=15):
    probs = F.softmax(logits, dim=-1)
    conf, preds = torch.max(probs, dim=1)
    acc = preds.eq(labels.to(device))
    ece = torch.zeros(1, device=device)
    for bin_lower in torch.linspace(0, 1, n_bins, device=device):
        bin_upper = bin_lower + 1/n_bins
        in_bin = (conf > bin_lower) & (conf <= bin_upper)
        if in_bin.any():
            ece += torch.abs(acc[in_bin].float().mean() - conf[in_bin].mean()) * in_bin.float().mean()
    return ece.item()

for field in label_spaces:
    logits = field_logits[field]
    y_true = targets[field]
    before = compute_ece(logits, y_true)
    T = calibrate(logits, y_true)
    after = compute_ece(logits / T, y_true)
    print(f"{field}: T={T:.2f}  |  ECE before {before:.3f} → after {after:.3f}")
