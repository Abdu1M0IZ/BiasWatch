import os
from pathlib import Path


APP_NAME = "BiasWatch Hate Speech Detection API"
APP_VERSION = "1.0.0"

DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "best")

MODEL_PATHS = {
    "best": Path(
        os.getenv(
            "MODEL_PATH",
            "artifacts/best_tuned_model.joblib",
        )
    )
}

LABEL_NAMES = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither",
}

ALLOWED_MODELS = set(MODEL_PATHS.keys())

CSV_TEXT_COLUMN_CANDIDATES = [
    "tweet",
    "text",
    "cleaned_tweet",
    "content",
    "message",
]

MAX_BATCH_SIZE = 100
MAX_CSV_ROWS = 1000
MAX_TEXT_LENGTH = 1000
MIN_TEXT_LENGTH = 3