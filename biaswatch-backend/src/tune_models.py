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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATASET_PATH = Path("data/processed/processed_dataset.csv")

ARTIFACTS_DIR = Path("artifacts")
REPORT_DIR = Path("report")

BEST_MODEL_PATH = ARTIFACTS_DIR / "best_tuned_model.joblib"
TUNING_RESULTS_PATH = ARTIFACTS_DIR / "tuning_results.csv"
TUNING_METRICS_PATH = ARTIFACTS_DIR / "tuning_metrics.json"
TUNING_REPORT_PATH = REPORT_DIR / "tuning_report.txt"
TUNING_ERRORS_PATH = REPORT_DIR / "tuning_error_analysis.csv"

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
    df["class"] = pd.to_numeric(df["class"], errors="coerce")

    df = df.dropna(subset=["class"]).copy()
    df["class"] = df["class"].astype(int)

    df = df[df["class"].isin([0, 1, 2])].copy()

    return df


def split_dataset(df):
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


def build_search_spaces():
    search_spaces = {}

    logreg_pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    logreg_params = {
        "tfidf__max_features": [30000, 50000],
        "tfidf__ngram_range": [(1, 2)],
        "tfidf__min_df": [1, 2],
        "tfidf__sublinear_tf": [True],
        "model__C": [1.0, 2.0],
        "model__class_weight": ["balanced"],
    }

    search_spaces["logistic_regression"] = {
        "pipeline": logreg_pipeline,
        "params": logreg_params,
    }

    svm_pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            (
                "model",
                LinearSVC(
                    random_state=RANDOM_SEED,
                    max_iter=5000,
                ),
            ),
        ]
    )

    svm_params = {
        "tfidf__max_features": [30000, 50000],
        "tfidf__ngram_range": [(1, 2)],
        "tfidf__min_df": [1, 2],
        "tfidf__sublinear_tf": [True],
        "model__C": [1.0, 2.0],
        "model__class_weight": ["balanced"],
    }

    search_spaces["linear_svm"] = {
        "pipeline": svm_pipeline,
        "params": svm_params,
    }

    nb_pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ]
    )

    nb_params = {
        "tfidf__max_features": [30000, 50000],
        "tfidf__ngram_range": [(1, 2)],
        "tfidf__min_df": [1, 2],
        "tfidf__sublinear_tf": [True],
        "model__alpha": [0.5, 1.0],
    }

    search_spaces["naive_bayes"] = {
        "pipeline": nb_pipeline,
        "params": nb_params,
    }

    return search_spaces


def run_grid_search(model_name, pipeline, params, x_train, y_train):
    print("\n" + "=" * 60)
    print(f"tuning {model_name}")
    print("=" * 60)

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
        error_score="raise",
    )

    search.fit(x_train, y_train)

    print(f"\nbest cv macro f1 for {model_name}: {search.best_score_:.4f}")
    print(f"best params for {model_name}:")
    print(search.best_params_)

    return search


def evaluate_on_validation(model, x_val, y_val):
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


def collect_search_results(model_name, search):
    results_df = pd.DataFrame(search.cv_results_)
    results_df["model_name"] = model_name

    selected_columns = [
        "model_name",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "rank_test_score",
        "params",
    ]

    return results_df[selected_columns].copy()


def save_error_analysis(x_val, y_val, y_pred):
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

    error_df.to_csv(TUNING_ERRORS_PATH, index=False)


def save_tuning_report(
    best_model_name,
    best_params,
    best_cv_score,
    validation_metrics,
    all_model_summaries,
):
    with open(TUNING_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("BiasWatch Hyperparameter Tuning Report\n")
        file.write("======================================\n\n")

        file.write("Dataset\n")
        file.write("-------\n")
        file.write(f"Dataset path: {DATASET_PATH}\n")
        file.write(f"Random seed: {RANDOM_SEED}\n")
        file.write(f"Validation size: {TEST_SIZE}\n")
        file.write("Split type: stratified train/validation split\n\n")

        file.write("Tuning Method\n")
        file.write("-------------\n")
        file.write(
            "GridSearchCV was used to brute-force multiple hyperparameter "
            "combinations for Logistic Regression, Linear SVM, and Naive Bayes. "
            "The scoring metric was macro-F1 because the dataset is imbalanced "
            "and all classes should be treated as important.\n\n"
        )

        file.write("Models Tuned\n")
        file.write("------------\n")

        for summary in all_model_summaries:
            file.write(f"Model: {summary['model_name']}\n")
            file.write(f"Best CV macro-F1: {summary['best_cv_macro_f1']:.4f}\n")
            file.write(f"Best params: {summary['best_params']}\n\n")

        file.write("Best Model\n")
        file.write("----------\n")
        file.write(f"Best model name: {best_model_name}\n")
        file.write(f"Best CV macro-F1: {best_cv_score:.4f}\n")
        file.write(f"Best parameters: {best_params}\n\n")

        file.write("Validation Results of Best Model\n")
        file.write("--------------------------------\n")
        file.write(f"Accuracy: {validation_metrics['accuracy']:.4f}\n")
        file.write(f"Macro F1: {validation_metrics['macro_f1']:.4f}\n")
        file.write(f"Weighted F1: {validation_metrics['weighted_f1']:.4f}\n")
        file.write(f"Macro Precision: {validation_metrics['macro_precision']:.4f}\n")
        file.write(f"Macro Recall: {validation_metrics['macro_recall']:.4f}\n\n")

        file.write("Confusion Matrix\n")
        file.write("----------------\n")
        file.write(json.dumps(validation_metrics["confusion_matrix"], indent=4))
        file.write("\n\n")

        file.write("Classification Report\n")
        file.write("---------------------\n")
        file.write(validation_metrics["classification_report_text"])
        file.write("\n\n")

        file.write("Leakage Control\n")
        file.write("---------------\n")
        file.write(
            "The validation set was kept separate from the tuning process. "
            "GridSearchCV was applied only on the training split. "
            "Inside each cross-validation fold, the TF-IDF vectorizer was fitted "
            "only on that fold's training portion through the scikit-learn pipeline. "
            "The final validation set was used only once for final evaluation.\n\n"
        )

        file.write("Engineering Note\n")
        file.write("----------------\n")
        file.write(
            "A higher number of hyperparameter combinations can improve the chance "
            "of finding a better model, but it also increases training time. "
            "Macro-F1 was prioritized over accuracy because the moderation task "
            "requires reasonable performance across hate_speech, offensive_language, "
            "and neither.\n"
        )


def save_metrics_json(
    best_model_name,
    best_params,
    best_cv_score,
    validation_metrics,
    all_model_summaries,
):
    metrics_copy = validation_metrics.copy()
    metrics_copy.pop("classification_report_text")

    output = {
        "best_model_name": best_model_name,
        "best_params": best_params,
        "best_cv_macro_f1": best_cv_score,
        "validation_metrics": metrics_copy,
        "all_model_summaries": all_model_summaries,
    }

    with open(TUNING_METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)


def tune_models():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    print("dataset loaded")
    print(f"total rows: {len(df)}")
    print("\nclass distribution:")
    print(df["class"].value_counts().sort_index())

    x_train, x_val, y_train, y_val = split_dataset(df)

    print("\ntrain/validation split created")
    print(f"training rows: {len(x_train)}")
    print(f"validation rows: {len(x_val)}")

    search_spaces = build_search_spaces()

    all_results = []
    all_model_summaries = []

    best_model_name = None
    best_model = None
    best_params = None
    best_cv_score = -1

    for model_name, setup in search_spaces.items():
        search = run_grid_search(
            model_name=model_name,
            pipeline=setup["pipeline"],
            params=setup["params"],
            x_train=x_train,
            y_train=y_train,
        )

        result_df = collect_search_results(model_name, search)
        all_results.append(result_df)

        model_summary = {
            "model_name": model_name,
            "best_cv_macro_f1": search.best_score_,
            "best_params": search.best_params_,
        }

        all_model_summaries.append(model_summary)

        model_path = ARTIFACTS_DIR / f"best_{model_name}.joblib"
        joblib.dump(search.best_estimator_, model_path)

        print(f"saved best {model_name} model to {model_path}")

        if search.best_score_ > best_cv_score:
            best_model_name = model_name
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = search.best_score_

    tuning_results_df = pd.concat(all_results, ignore_index=True)
    tuning_results_df = tuning_results_df.sort_values(
        by="mean_test_score",
        ascending=False,
    )

    tuning_results_df.to_csv(TUNING_RESULTS_PATH, index=False)

    validation_metrics, y_pred = evaluate_on_validation(
        best_model,
        x_val,
        y_val,
    )

    joblib.dump(best_model, BEST_MODEL_PATH)

    save_error_analysis(x_val, y_val, y_pred)

    save_metrics_json(
        best_model_name=best_model_name,
        best_params=best_params,
        best_cv_score=best_cv_score,
        validation_metrics=validation_metrics,
        all_model_summaries=all_model_summaries,
    )

    save_tuning_report(
        best_model_name=best_model_name,
        best_params=best_params,
        best_cv_score=best_cv_score,
        validation_metrics=validation_metrics,
        all_model_summaries=all_model_summaries,
    )

    print("\ntuning complete")
    print("---------------")
    print(f"best model: {best_model_name}")
    print(f"best cv macro f1: {best_cv_score:.4f}")
    print(f"validation accuracy: {validation_metrics['accuracy']:.4f}")
    print(f"validation macro f1: {validation_metrics['macro_f1']:.4f}")
    print(f"validation weighted f1: {validation_metrics['weighted_f1']:.4f}")
    print(f"validation macro precision: {validation_metrics['macro_precision']:.4f}")
    print(f"validation macro recall: {validation_metrics['macro_recall']:.4f}")

    print(f"\nsaved best model to {BEST_MODEL_PATH}")
    print(f"saved tuning results to {TUNING_RESULTS_PATH}")
    print(f"saved tuning metrics to {TUNING_METRICS_PATH}")
    print(f"saved tuning report to {TUNING_REPORT_PATH}")
    print(f"saved error analysis to {TUNING_ERRORS_PATH}")



tune_models()