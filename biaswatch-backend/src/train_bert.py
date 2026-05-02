from pathlib import Path
import json
import os
import random

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DATASET_PATH = Path("data/processed/processed_dataset.csv")

ARTIFACTS_DIR = Path("artifacts")
BERT_MODEL_DIR = ARTIFACTS_DIR / "bert_model"
BERT_METRICS_PATH = ARTIFACTS_DIR / "bert_metrics.json"
BERT_REPORT_PATH = Path("report/bert_results.txt")

RANDOM_SEED = 42
TEST_SIZE = 0.2

# Practical default for local training.
# For full BERT, change this to: "bert-base-uncased"
MODEL_CHECKPOINT = os.getenv("HF_MODEL_NAME", "distilbert-base-uncased")

MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", "128"))
NUM_EPOCHS = float(os.getenv("BERT_EPOCHS", "2"))
TRAIN_BATCH_SIZE = int(os.getenv("BERT_TRAIN_BATCH_SIZE", "8"))
EVAL_BATCH_SIZE = int(os.getenv("BERT_EVAL_BATCH_SIZE", "16"))
LEARNING_RATE = float(os.getenv("BERT_LEARNING_RATE", "2e-5"))

LABEL_NAMES = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither",
}

ID2LABEL = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither",
}

LABEL2ID = {
    "hate_speech": 0,
    "offensive_language": 1,
    "neither": 2,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"cleaned_tweet", "class"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.dropna(subset=["cleaned_tweet", "class"]).copy()

    df["cleaned_tweet"] = df["cleaned_tweet"].astype(str)
    df["class"] = pd.to_numeric(df["class"], errors="coerce")

    df = df.dropna(subset=["class"]).copy()
    df["class"] = df["class"].astype(int)

    df = df[df["class"].isin([0, 1, 2])].copy()

    df = df.rename(
        columns={
            "cleaned_tweet": "text",
            "class": "label",
        }
    )

    df = df[["text", "label"]].copy()

    return df


def create_splits(df):
    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    return train_df, val_df


def build_hf_datasets(train_df, val_df):
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    return train_dataset, val_dataset


def tokenize_datasets(train_dataset, val_dataset, tokenizer):
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    val_dataset = val_dataset.map(tokenize_batch, batched=True)

    return train_dataset, val_dataset


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
        "macro_precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }


def evaluate_final_model(trainer, val_dataset, val_df):
    raw_predictions = trainer.predict(val_dataset)
    logits = raw_predictions.predictions
    predictions = np.argmax(logits, axis=-1)
    true_labels = val_df["label"].values

    metrics = {
        "model_checkpoint": MODEL_CHECKPOINT,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "max_length": MAX_LENGTH,
        "num_epochs": NUM_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "accuracy": accuracy_score(true_labels, predictions),
        "macro_f1": f1_score(true_labels, predictions, average="macro"),
        "weighted_f1": f1_score(true_labels, predictions, average="weighted"),
        "macro_precision": precision_score(true_labels, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(true_labels, predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(true_labels, predictions).tolist(),
        "classification_report": classification_report(
            true_labels,
            predictions,
            target_names=[
                LABEL_NAMES[0],
                LABEL_NAMES[1],
                LABEL_NAMES[2],
            ],
            output_dict=True,
            zero_division=0,
        ),
        "classification_report_text": classification_report(
            true_labels,
            predictions,
            target_names=[
                LABEL_NAMES[0],
                LABEL_NAMES[1],
                LABEL_NAMES[2],
            ],
            zero_division=0,
        ),
    }

    return metrics, predictions


def save_metrics(metrics):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    BERT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    metrics_json = metrics.copy()
    metrics_json.pop("classification_report_text")

    with open(BERT_METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics_json, file, indent=4)

    with open(BERT_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("BiasWatch BERT Model Evaluation Report\n")
        file.write("=====================================\n\n")

        file.write("Model Configuration\n")
        file.write("-------------------\n")
        file.write(f"Model checkpoint: {metrics['model_checkpoint']}\n")
        file.write(f"Max length: {metrics['max_length']}\n")
        file.write(f"Epochs: {metrics['num_epochs']}\n")
        file.write(f"Learning rate: {metrics['learning_rate']}\n")
        file.write(f"Train batch size: {metrics['train_batch_size']}\n")
        file.write(f"Eval batch size: {metrics['eval_batch_size']}\n\n")

        file.write("Evaluation Metrics\n")
        file.write("------------------\n")
        file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        file.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")
        file.write(f"Weighted F1: {metrics['weighted_f1']:.4f}\n")
        file.write(f"Macro Precision: {metrics['macro_precision']:.4f}\n")
        file.write(f"Macro Recall: {metrics['macro_recall']:.4f}\n\n")

        file.write("Confusion Matrix\n")
        file.write("----------------\n")
        file.write(json.dumps(metrics["confusion_matrix"], indent=4))
        file.write("\n\n")

        file.write("Classification Report\n")
        file.write("---------------------\n")
        file.write(metrics["classification_report_text"])


def train_bert():
    set_seed(RANDOM_SEED)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset()

    print(f"Total rows: {len(df)}")
    print("Class distribution:")
    print(df["label"].value_counts().sort_index())

    train_df, val_df = create_splits(df)

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")

    train_dataset, val_dataset = build_hf_datasets(train_df, val_df)

    print(f"Loading tokenizer: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    train_dataset, val_dataset = tokenize_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"Loading model: {MODEL_CHECKPOINT}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(ARTIFACTS_DIR / "bert_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=str(ARTIFACTS_DIR / "bert_logs"),
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training BERT-family model...")
    trainer.train()

    print("Evaluating final model...")
    metrics, predictions = evaluate_final_model(
        trainer=trainer,
        val_dataset=val_dataset,
        val_df=val_df,
    )

    print("Saving model and tokenizer...")
    trainer.save_model(str(BERT_MODEL_DIR))
    tokenizer.save_pretrained(str(BERT_MODEL_DIR))

    save_metrics(metrics)

    print("\nBERT training complete")
    print("----------------------")
    print(f"Saved model to: {BERT_MODEL_DIR}")
    print(f"Saved metrics to: {BERT_METRICS_PATH}")
    print(f"Saved report to: {BERT_REPORT_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")


if __name__ == "__main__":
    train_bert()