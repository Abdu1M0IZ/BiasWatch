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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATASET_PATH = Path("data/processed/processed_dataset.csv")

ARTIFACTS_DIR = Path("artifacts")
REPORT_DIR = Path("report")

METRICS_PATH = ARTIFACTS_DIR / "all_model_metrics.json"
SUMMARY_REPORT_PATH = REPORT_DIR / "assignment2_results.txt"
ERROR_ANALYSIS_PATH = REPORT_DIR / "error_analysis.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.2

LABEL_NAMES = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither",
}


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

    df = df[df["class"].isin([0, 1, 2])].copy()

    return df


def split_data(df):
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


def build_model_configs():
    configs = {
        "logreg_basic": Pipeline(
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
        ),
        "logreg_tuned": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=80000,
                        ngram_range=(1, 2),
                        min_df=2,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=2.0,
                        max_iter=1500,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=80000,
                        ngram_range=(1, 2),
                        min_df=2,
                    ),
                ),
                (
                    "model",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "naive_bayes": Pipeline(
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
                    MultinomialNB(),
                ),
            ]
        ),
    }

    return configs


def evaluate_model(model, x_val, y_val):
    y_pred = model.predict(x_val)

    target_names = [
        LABEL_NAMES[0],
        LABEL_NAMES[1],
        LABEL_NAMES[2],
    ]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "macro_f1": f1_score(y_val, y_pred, average="macro"),
        "weighted_f1": f1_score(y_val, y_pred, average="weighted"),
        "macro_precision": precision_score(y_val, y_pred, average="macro"),
        "macro_recall": recall_score(y_val, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
        "classification_report": classification_report(
            y_val,
            y_pred,
            target_names=target_names,
            output_dict=True,
        ),
        "classification_report_text": classification_report(
            y_val,
            y_pred,
            target_names=target_names,
        ),
    }

    return metrics, y_pred


def save_error_analysis(x_val, y_val, y_pred, best_model_name):
    error_df = pd.DataFrame(
        {
            "cleaned_tweet": x_val.values,
            "true_label": y_val.values,
            "predicted_label": y_pred,
        }
    )

    error_df["true_label_name"] = error_df["true_label"].map(LABEL_NAMES)
    error_df["predicted_label_name"] = error_df["predicted_label"].map(LABEL_NAMES)

    error_df = error_df[error_df["true_label"] != error_df["predicted_label"]]

    error_df.to_csv(ERROR_ANALYSIS_PATH, index=False)

    print(f"saved error analysis for {best_model_name} to {ERROR_ANALYSIS_PATH}")


def save_summary_report(results, best_model_name):
    with open(SUMMARY_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("Dataset\n")
        file.write("-------\n")
        file.write(f"Dataset path: {DATASET_PATH}\n")
        file.write(f"Random seed: {RANDOM_SEED}\n")
        file.write(f"Validation size: {TEST_SIZE}\n")
        file.write("Stratified split: yes\n\n")

        file.write("Leakage Control\n")
        file.write("---------------\n")
        file.write(
            "The train/validation split was performed before fitting TF-IDF. "
            "Each model pipeline fitted TF-IDF only on the training split. "
            "The validation split was only transformed using the fitted vectorizer.\n\n"
        )

        file.write("Model Results\n")
        file.write("-------------\n")

        for model_name, metrics in results.items():
            file.write(f"\nModel: {model_name}\n")
            file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            file.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")
            file.write(f"Weighted F1: {metrics['weighted_f1']:.4f}\n")
            file.write(f"Macro Precision: {metrics['macro_precision']:.4f}\n")
            file.write(f"Macro Recall: {metrics['macro_recall']:.4f}\n")
            file.write("Confusion Matrix:\n")
            file.write(json.dumps(metrics["confusion_matrix"], indent=4))
            file.write("\n")

        file.write("\nBest Model\n")
        file.write("----------\n")
        file.write(f"Best model by macro-F1: {best_model_name}\n\n")

        file.write("Interpretation Notes\n")
        file.write("--------------------\n")
        file.write(
            "Accuracy is not enough for this dataset because the classes are imbalanced. "
            "Macro-F1 is used as the main comparison metric because it gives equal importance "
            "to hate_speech, offensive_language, and neither. Linear SVM is expected to perform "
            "well because margin-based classifiers usually work strongly with sparse TF-IDF features. "
            "Logistic Regression is easier to interpret and can provide probabilities. "
            "Naive Bayes is very fast but usually less flexible than Logistic Regression or SVM.\n"
        )


def save_metrics_json(results):
    json_ready_results = {}

    for model_name, metrics in results.items():
        metrics_copy = metrics.copy()
        metrics_copy.pop("classification_report_text")
        json_ready_results[model_name] = metrics_copy

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(json_ready_results, file, indent=4)


def train_all_models():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    x_train, x_val, y_train, y_val = split_data(df)

    model_configs = build_model_configs()

    results = {}
    predictions = {}

    for model_name, model in model_configs.items():
        print(f"\ntraining {model_name}...")

        model.fit(x_train, y_train)

        metrics, y_pred = evaluate_model(model, x_val, y_val)

        results[model_name] = metrics
        predictions[model_name] = y_pred

        model_path = ARTIFACTS_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        print(f"saved model to {model_path}")
        print(f"accuracy: {metrics['accuracy']:.4f}")
        print(f"macro f1: {metrics['macro_f1']:.4f}")
        print(f"weighted f1: {metrics['weighted_f1']:.4f}")

    best_model_name = max(
        results,
        key=lambda name: results[name]["macro_f1"],
    )

    save_error_analysis(
        x_val,
        y_val,
        predictions[best_model_name],
        best_model_name,
    )

    save_metrics_json(results)
    save_summary_report(results, best_model_name)

    print(f"best model by macro f1: {best_model_name}")
    print(f"saved metrics to {METRICS_PATH}")
    print(f"saved summary report to {SUMMARY_REPORT_PATH}")
    print(f"saved error analysis to {ERROR_ANALYSIS_PATH}")


train_all_models()