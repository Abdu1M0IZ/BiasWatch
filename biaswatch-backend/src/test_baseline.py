from pathlib import Path

import joblib


MODEL_PATH = Path("artifacts/baseline_model.joblib")

LABEL_NAMES = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither",
}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def predict_examples():
    model = load_model()

    examples = [
        "i hate this",
        "i hope you have a great day",
        "this group should not exist",
        "what a beautiful morning",
        "shut up you idiot",
    ]

    predictions = model.predict(examples)

    for text, label_id in zip(examples, predictions):
        print("-" * 60)
        print(f"text: {text}")
        print(f"predicted class: {label_id}")
        print(f"label name: {LABEL_NAMES[label_id]}")



predict_examples()