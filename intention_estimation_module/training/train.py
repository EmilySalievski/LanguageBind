import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from dataset import load_intent_dataset

MODEL_NAME = "microsoft/deberta-v3-large"

def main():
    dataset = load_intent_dataset("data/intents.csv")
    labels = list(set(dataset["train"]["label"]) | set(dataset["test"]["label"]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    def preprocess(batch):
        enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
        enc["label"] = [label2id[l] for l in batch["label"]]
        return enc

    tokenized = dataset.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id, use_safetensors=True,
    )

    training_args = TrainingArguments(
        output_dir="checkpoints/",
        # evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="logs/",
        fp16=torch.cuda.is_available(),
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained("models/fine_tuned_classifier")
    tokenizer.save_pretrained("models/fine_tuned_classifier")

if __name__ == "__main__":
    main()
