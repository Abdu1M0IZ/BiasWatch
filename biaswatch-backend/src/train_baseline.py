from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATASET_PATH = Path("data/processed/processed_dataset.csv")
ARTIFACT_PATH = Path("artifacts/baseline_model.joblib")
METRICS_PATH = Path("artifacts/baseline_metrics.json")
REPORT_PATH = Path("report/baseline_results.txt")

RANDOM_SEED = 42
TEST_SIZE = 0.2

LABEL_NAMES = [
    "hate_speech",
    "offensive_language",
    "neither",
]


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"processed dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"cleaned_tweet", "class"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    df = df.dropna(subset=["cleaned_tweet", "class"]).copy()

    df["cleaned_tweet"] = df["cleaned_tweet"].astype(str)
    df["class"] = df["class"].astype(int)

    valid_labels = [0, 1, 2]
    df = df[df["class"].isin(valid_labels)].copy()

    return df


def create_train_validation_split(df):
    x = df["cleaned_tweet"]
    y = df["class"]

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    return x_train, x_val, y_train, y_val


def build_model_pipeline():
    model_pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    return model_pipeline


def evaluate_model(model_pipeline, x_val, y_val):
    y_pred = model_pipeline.predict(x_val)

    metrics = {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "validation_rows": len(x_val),
        "accuracy": accuracy_score(y_val, y_pred),
        "macro_f1": f1_score(y_val, y_pred, average="macro"),
        "weighted_f1": f1_score(y_val, y_pred, average="weighted"),
        "macro_precision": precision_score(y_val, y_pred, average="macro"),
        "macro_recall": recall_score(y_val, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
        "classification_report": classification_report(
            y_val,
            y_pred,
            target_names=LABEL_NAMES,
            output_dict=True,
        ),
        "classification_report_text": classification_report(
            y_val,
            y_pred,
            target_names=LABEL_NAMES,
        ),
    }

    return metrics


def save_artifacts(model_pipeline, metrics):
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_pipeline, ARTIFACT_PATH)

    metrics_for_json = metrics.copy()
    metrics_for_json.pop("classification_report_text")

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics_for_json, file, indent=4)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("Model\n")
        file.write("-----\n")
        file.write("TF-IDF Vectorizer + Logistic Regression\n\n")

        file.write("Train/Validation Split\n")
        file.write("----------------------\n")
        file.write(f"Test size: {TEST_SIZE}\n")
        file.write(f"Random seed: {RANDOM_SEED}\n")
        file.write(f"Validation rows: {metrics['validation_rows']}\n")
        file.write("Stratified split: yes\n\n")

        file.write("Metrics\n")
        file.write("-------\n")
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

    print(f"saved model to {ARTIFACT_PATH}")
    print(f"saved metrics to {METRICS_PATH}")
    print(f"saved baseline report to {REPORT_PATH}")



df = load_dataset()

x_train, x_val, y_train, y_val = create_train_validation_split(df)

model_pipeline = build_model_pipeline()

model_pipeline.fit(x_train, y_train)

metrics = evaluate_model(model_pipeline, x_val, y_val)

save_artifacts(model_pipeline, metrics)

print("\nBaseline model results")
print("----------------------")
print(f"accuracy: {metrics['accuracy']:.4f}")
print(f"macro f1: {metrics['macro_f1']:.4f}")
print(f"weighted f1: {metrics['weighted_f1']:.4f}")
print(f"macro precision: {metrics['macro_precision']:.4f}")
print(f"macro recall: {metrics['macro_recall']:.4f}")

