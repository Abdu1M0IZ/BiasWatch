from pathlib import Path

import joblib


MODEL_PATH = Path("artifacts/best_tuned_model.joblib")

LABEL_NAMES = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither",
}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"best tuned model not found: {MODEL_PATH}. run src/tune_models.py first."
        )

    return joblib.load(MODEL_PATH)


def test_examples():
    model = load_model()

    examples = [
        "you are so stupid",
        "hope you have a great day",
        "this is a normal update about my morning",
        "shut up you idiot",
        "i strongly disagree with your opinion",
    ]

    predictions = model.predict(examples)

    for text, label_id in zip(examples, predictions):
        print("-" * 60)
        print(f"text: {text}")
        print(f"predicted label id: {label_id}")
        print(f"predicted label name: {LABEL_NAMES[label_id]}")



test_examples()