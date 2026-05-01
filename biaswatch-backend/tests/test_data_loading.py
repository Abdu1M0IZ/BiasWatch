from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/processed/processed_dataset.csv")


def test_processed_dataset_exists():
    assert DATASET_PATH.exists()


def test_processed_dataset_has_required_columns():
    df = pd.read_csv(DATASET_PATH)

    required_columns = {
        "tweet",
        "class",
        "source",
        "cleaned_tweet",
        "label_name",
    }

    assert required_columns.issubset(set(df.columns))


def test_processed_dataset_has_valid_labels():
    df = pd.read_csv(DATASET_PATH)

    valid_labels = {0, 1, 2}
    dataset_labels = set(df["class"].unique())

    assert dataset_labels.issubset(valid_labels)


def test_processed_dataset_has_no_empty_cleaned_tweets():
    df = pd.read_csv(DATASET_PATH)

    empty_count = df["cleaned_tweet"].isna().sum()
    empty_count += (df["cleaned_tweet"].astype(str).str.strip() == "").sum()

    assert empty_count == 0