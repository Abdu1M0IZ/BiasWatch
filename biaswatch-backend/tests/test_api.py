from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_200():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_returns_best_model():
    with TestClient(app) as client:
        response = client.get("/models")

    data = response.json()

    assert response.status_code == 200
    assert "best" in data["available_models"]


def test_predict_single_text():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "text": "you are so stupid",
                "model_name": "best",
            },
        )

    data = response.json()

    assert response.status_code == 200
    assert "label_id" in data
    assert "label_name" in data
    assert "confidence" in data
    assert "scores" in data
    assert "cleaned_text" in data


def test_predict_rejects_blank_text():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "text": "   ",
                "model_name": "best",
            },
        )

    assert response.status_code == 422


def test_predict_rejects_extra_fields():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "text": "hello world",
                "model_name": "best",
                "extra": "not allowed",
            },
        )

    assert response.status_code == 422


def test_batch_predict():
    with TestClient(app) as client:
        response = client.post(
            "/batch-predict",
            json={
                "texts": [
                    "you are stupid",
                    "hope you have a nice day",
                ],
                "model_name": "best",
            },
        )

    data = response.json()

    assert response.status_code == 200
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_invalid_model_name_fails():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "text": "hello world",
                "model_name": "unknown",
            },
        )

    assert response.status_code == 422


def test_csv_prediction_with_tweet_column():
    csv_content = "tweet\nyou are stupid\nhope you have a nice day\n"

    with TestClient(app) as client:
        response = client.post(
            "/predict-csv",
            files={
                "file": (
                    "sample.csv",
                    csv_content,
                    "text/csv",
                )
            },
            data={
                "model_name": "best",
            },
        )

    data = response.json()

    assert response.status_code == 200
    assert data["detected_text_column"] == "tweet"
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_csv_prediction_with_custom_column():
    csv_content = "post\nhello there\nthis is a normal message\n"

    with TestClient(app) as client:
        response = client.post(
            "/predict-csv",
            files={
                "file": (
                    "sample.csv",
                    csv_content,
                    "text/csv",
                )
            },
            data={
                "model_name": "best",
                "text_column": "post",
            },
        )

    data = response.json()

    assert response.status_code == 200
    assert data["detected_text_column"] == "post"
    assert data["count"] == 2