import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from dataset import load_intent_dataset_jsonl

MODEL_NAME = "microsoft/deberta-v3-large"

def main():
    dataset = load_intent_dataset_jsonl("data/ground_truth.jsonl")

    # Extract unique labels dynamically
    labels = sorted(list(set(dataset["train"]["label"]) | set(dataset["test"]["label"])))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    def preprocess(batch):
        enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
        enc["label"] = [label2id[l] for l in batch["label"]]
        return enc

    tokenized = dataset.map(preprocess, batched=True, remove_columns=["text", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_proj", "key_proj", "value_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="checkpoints/",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="logs/",
        fp16=torch.cuda.is_available(),
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels_eval = eval_pred
        preds = logits.argmax(axis=-1)
        return metric.compute(predictions=preds, references=labels_eval)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained("models/fine_tuned_classifier", safe_serialization=True)
    tokenizer.save_pretrained("models/fine_tuned_classifier")

if __name__ == "__main__":
    main()


# import torch
# import evaluate
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from peft import LoraConfig, get_peft_model
# from dataset import load_intent_dataset_jsonl
#
# MODEL_NAME = "microsoft/deberta-v3-large"
#
# def main():
#     dataset = load_intent_dataset_jsonl("data/ground_truth.jsonl")
#
#     # Extract unique labels
#     labels = list(set(dataset["train"]["label"]) | set(dataset["test"]["label"]))
#     label2id = {l: i for i, l in enumerate(labels)}
#     id2label = {i: l for l, i in label2id.items()}
#
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
#
#     def preprocess(batch):
#         enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
#         enc["label"] = [label2id[l] for l in batch["label"]]
#         return enc
#
#     tokenized = dataset.map(preprocess, batched=True, remove_columns=["text", "label"])
#
#     # Load base model
#     model = AutoModelForSequenceClassification.from_pretrained(
#         MODEL_NAME,
#         num_labels=len(labels),
#         id2label=id2label,
#         label2id=label2id,
#         use_safetensors=True
#     )
#
#     for name, module in model.named_modules():
#         if "attention" in name.lower() or "query" in name.lower():
#             print(name, module)
#
#     # Apply LoRA
#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules=["query_proj", "key_proj", "value_proj"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type="SEQ_CLS"
#     )
#     model = get_peft_model(model, lora_config)
#
#     training_args = TrainingArguments(
#         output_dir="checkpoints/",
#         save_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=4,  # reduce if GPU VRAM is limited
#         per_device_eval_batch_size=4,
#         num_train_epochs=5,
#         weight_decay=0.01,
#         logging_dir="logs/",
#         fp16=torch.cuda.is_available(),
#     )
#
#     metric = evaluate.load("accuracy")
#
#     def compute_metrics(eval_pred):
#         logits, labels_eval = eval_pred
#         preds = logits.argmax(axis=-1)
#         return metric.compute(predictions=preds, references=labels_eval)
#
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized["train"],
#         eval_dataset=tokenized["test"],
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics,
#     )
#
#     trainer.train()
#     model.save_pretrained("models/fine_tuned_classifier", safe_serialization=True)
#     tokenizer.save_pretrained("models/fine_tuned_classifier")
#
# if __name__ == "__main__":
#     main()