from pathlib import Path
import html
import re

import pandas as pd

RAW_DATA_PATH = Path("data/raw/")
PROCESSED_DATA_PATH = Path("data/processed/")
REPORT_PATH = Path("report/")

DAVIDSON_DATASET_PATH = RAW_DATA_PATH / "Davidson_dataset.csv"
HATEEXT_DATASET_PATH = RAW_DATA_PATH / "hatexplain.csv"

OUTPUT_DATASET_PATH = PROCESSED_DATA_PATH / "processed_dataset.csv"
REPORT_OUTPUT_PATH = REPORT_PATH / "proprocessing_results.txt"

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
RT_PATTERN = re.compile(r"\brt\b", flags=re.IGNORECASE)
SPACE_PATTERN = re.compile(r"\s+")

LABEL_NAMES = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither",
}


def clean_text(text):
    text = str(text)
    text = html.unescape(text)
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = RT_PATTERN.sub(" ", text)
    text = re.sub(r"[^a-z0-9#' ]+", " ", text)
    text = SPACE_PATTERN.sub(" ", text).strip()
    return text


def load_dataset(path, source_name):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)

    unnamed_cols = [
        col for col in df.columns
        if col.lower().startswith("unnamed")
    ]

    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_columns = {"tweet", "class"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {path}: {missing_columns}"
        )

    df = df.copy()
    df["source"] = source_name

    return df


def standardize_dataset(df):
    df = df.copy()

    df = df[["tweet", "class", "source"]]

    df = df.dropna(subset=["tweet", "class"])

    df["tweet"] = df["tweet"].astype(str)
    df["class"] = pd.to_numeric(df["class"], errors="coerce")

    df = df.dropna(subset=["class"])
    df["class"] = df["class"].astype(int)

    valid_labels = [0, 1, 2]
    df = df[df["class"].isin(valid_labels)].copy()

    return df


def merge_datasets():
    davidson_df = load_dataset(DAVIDSON_DATASET_PATH, "davidson")
    hatexplain_df = load_dataset(HATEEXT_DATASET_PATH, "hatexplain")

    davidson_df = standardize_dataset(davidson_df)
    hatexplain_df = standardize_dataset(hatexplain_df)

    merged_df = pd.concat(
        [davidson_df, hatexplain_df],
        ignore_index=True
    )

    return merged_df


def preprocess(df):
    df = df.copy()

    before_empty_removal = len(df)

    df["cleaned_tweet"] = df["tweet"].apply(clean_text)

    df = df[df["cleaned_tweet"].str.len() > 0].copy()

    empty_rows_removed = before_empty_removal - len(df)

    df["label_name"] = df["class"].map(LABEL_NAMES)

    duplicate_conflicts = (
        df.groupby("cleaned_tweet")["class"]
        .nunique()
        .reset_index(name="num_labels")
    )

    conflicting_texts = duplicate_conflicts[
        duplicate_conflicts["num_labels"] > 1
    ]["cleaned_tweet"]

    if len(conflicting_texts) > 0:
        print(
            f"Warning: {len(conflicting_texts)} cleaned tweets have conflicting labels."
        )

    before_duplicate_removal = len(df)

    df = df.drop_duplicates(
        subset=["cleaned_tweet", "class"],
        keep="first"
    ).copy()

    duplicate_rows_removed = before_duplicate_removal - len(df)

    df = df.reset_index(drop=True)

    preprocessing_info = {
        "empty_rows_removed": empty_rows_removed,
        "duplicate_rows_removed": duplicate_rows_removed,
        "conflicting_cleaned_tweets": len(conflicting_texts),
    }

    return df, preprocessing_info


def save_dataset(df):
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DATASET_PATH, index=False)
    print(f"Saved processed dataset to {OUTPUT_DATASET_PATH}")


def save_report(merged_df, processed_df, preprocessing_info):
    REPORT_PATH.mkdir(parents=True, exist_ok=True)

    source_counts_before = merged_df["source"].value_counts()
    source_counts_after = processed_df["source"].value_counts()

    class_counts = processed_df["class"].value_counts().sort_index()
    label_counts = processed_df["label_name"].value_counts()

    tweet_lengths = processed_df["cleaned_tweet"].str.len()

    total_rows = len(processed_df)

    class_percentages = (class_counts / total_rows * 100).round(2)

    report_text = f"""
Final Processed Dataset
------------------------
Processed dataset path: {OUTPUT_DATASET_PATH}

Rows after preprocessing:
{source_counts_after.to_string()}

Total rows after preprocessing: {len(processed_df)}

Final Class Distribution
-------------------------
Class ID counts:
{class_counts.to_string()}

Label name counts:
{label_counts.to_string()}

Class percentages:
{class_percentages.to_string()}

Label mapping:
0 = hate_speech
1 = offensive_language
2 = neither

Tweet Length Statistics
-------------------------
Minimum length: {tweet_lengths.min()}
Maximum length: {tweet_lengths.max()}
Mean length: {round(tweet_lengths.mean(), 2)}
Median length: {tweet_lengths.median()}
"""

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write(report_text.strip())

    print(f"Saved preprocessing report to {REPORT_OUTPUT_PATH}")


def main():
    merged_df = merge_datasets()
    processed_df, preprocessing_info = preprocess(merged_df)

    save_dataset(processed_df)
    save_report(merged_df, processed_df, preprocessing_info)


if __name__ == "__main__":
    main()