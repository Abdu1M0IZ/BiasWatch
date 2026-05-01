from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np

from app.config import LABEL_NAMES, MODEL_PATHS
from src.preprocessor import clean_text


def softmax_2d(values):
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    values = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(values)

    return exp_values / exp_values.sum(axis=1, keepdims=True)


class ModelStore:
    def __init__(self):
        self.models = {}

    def load_models(self):
        for model_name, model_path in MODEL_PATHS.items():
            model_path = Path(model_path)

            if not model_path.exists():
                raise FileNotFoundError(f"model file not found: {model_path}")

            self.models[model_name] = joblib.load(model_path)

    def get_loaded_model_names(self):
        return sorted(self.models.keys())

    def get_model(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"unknown model: {model_name}")

        return self.models[model_name]

    def predict_one(self, text, model_name="best", row_index: Optional[int] = None):
        predictions = self.predict_many(
            texts=[text],
            model_name=model_name,
            row_indexes=[row_index],
        )

        return predictions[0]

    def predict_many(self, texts: List[str], model_name="best", row_indexes=None):
        model = self.get_model(model_name)

        cleaned_texts = []
        original_texts = []
        valid_row_indexes = []

        if row_indexes is None:
            row_indexes = [None] * len(texts)

        for text, row_index in zip(texts, row_indexes):
            cleaned_text = clean_text(text)

            if not cleaned_text:
                raise ValueError("one or more texts became empty after cleaning")

            cleaned_texts.append(cleaned_text)
            original_texts.append(text)
            valid_row_indexes.append(row_index)

        label_ids = model.predict(cleaned_texts)
        score_rows = self.get_score_rows(model, cleaned_texts, label_ids)

        predictions = []

        for original_text, cleaned_text, label_id, scores, row_index in zip(
            original_texts,
            cleaned_texts,
            label_ids,
            score_rows,
            valid_row_indexes,
        ):
            label_id = int(label_id)
            label_name = LABEL_NAMES[label_id]
            confidence = float(scores[label_name])

            predictions.append(
                {
                    "row_index": row_index,
                    "model_name": model_name,
                    "input_text": str(original_text),
                    "cleaned_text": cleaned_text,
                    "label_id": label_id,
                    "label_name": label_name,
                    "confidence": round(confidence, 4),
                    "scores": {
                        label: round(float(score), 4)
                        for label, score in scores.items()
                    },
                }
            )

        return predictions

    def get_score_rows(self, model, cleaned_texts, label_ids):
        probabilities = None
        classes = None

        try:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(cleaned_texts)
                classes = self.get_model_classes(model, probabilities.shape[1])

        except Exception:
            probabilities = None
            classes = None

        if probabilities is None:
            try:
                if hasattr(model, "decision_function"):
                    decision_scores = model.decision_function(cleaned_texts)
                    probabilities = softmax_2d(decision_scores)
                    classes = self.get_model_classes(model, probabilities.shape[1])

            except Exception:
                probabilities = None
                classes = None

        if probabilities is None:
            probabilities = np.zeros((len(cleaned_texts), len(LABEL_NAMES)))

            for index, label_id in enumerate(label_ids):
                probabilities[index, int(label_id)] = 1.0

            classes = list(LABEL_NAMES.keys())

        score_rows = []

        for row in probabilities:
            scores = {
                label_name: 0.0
                for label_name in LABEL_NAMES.values()
            }

            for class_id, score in zip(classes, row):
                class_id = int(class_id)

                if class_id in LABEL_NAMES:
                    scores[LABEL_NAMES[class_id]] = float(score)

            score_rows.append(scores)

        return score_rows

    def get_model_classes(self, model, number_of_scores):
        if hasattr(model, "classes_"):
            return [int(value) for value in model.classes_]

        if hasattr(model, "named_steps"):
            final_model = model.named_steps.get("model")

            if final_model is not None and hasattr(final_model, "classes_"):
                return [int(value) for value in final_model.classes_]

        return list(range(number_of_scores))


model_store = ModelStore()