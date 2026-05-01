from contextlib import asynccontextmanager
import io
import os

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    APP_NAME,
    APP_VERSION,
    CSV_TEXT_COLUMN_CANDIDATES,
    DEFAULT_MODEL_NAME,
    MAX_CSV_ROWS,
)
from app.model_store import model_store
from app.schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    CsvPredictionResponse,
    HealthResponse,
    ModelsResponse,
    PredictionResponse,
    PredictRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_store.load_models()
    yield


app = FastAPI(
    title=APP_NAME,
    description=(
        "BiasWatch classifies social media text into hate speech, "
        "offensive language, or neither."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    FRONTEND_URL,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["root"])
def root():
    return {
        "message": "BiasWatch API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return {
        "status": "ok",
        "loaded_models": model_store.get_loaded_model_names(),
    }


@app.get("/models", response_model=ModelsResponse, tags=["system"])
def models():
    return {
        "available_models": model_store.get_loaded_model_names(),
        "default_model": DEFAULT_MODEL_NAME,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(request: PredictRequest):
    try:
        return model_store.predict_one(
            text=request.text,
            model_name=request.model_name,
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed: {str(error)}",
        )


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["prediction"])
def batch_predict(request: BatchPredictRequest):
    try:
        predictions = model_store.predict_many(
            texts=request.texts,
            model_name=request.model_name,
        )

        return {
            "model_name": request.model_name,
            "count": len(predictions),
            "predictions": predictions,
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"batch prediction failed: {str(error)}",
        )


@app.post("/predict-csv", response_model=CsvPredictionResponse, tags=["prediction"])
async def predict_csv(
    file: UploadFile = File(...),
    model_name: str = Form(DEFAULT_MODEL_NAME),
    text_column: str | None = Form(None),
):
    try:
        validate_model_name(model_name)
        validate_csv_file(file)

        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail="uploaded CSV is empty")

        df = pd.read_csv(io.BytesIO(content))

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="CSV has no rows")

        if len(df) > MAX_CSV_ROWS:
            raise HTTPException(
                status_code=400,
                detail=f"CSV row limit exceeded. Maximum allowed rows: {MAX_CSV_ROWS}",
            )

        detected_column = detect_text_column(df, text_column)

        texts = df[detected_column].fillna("").astype(str).tolist()
        row_indexes = df.index.tolist()

        blank_count = sum(1 for value in texts if not value.strip())

        if blank_count > 0:
            raise HTTPException(
                status_code=400,
                detail=f"CSV contains {blank_count} blank text values in column '{detected_column}'",
            )

        predictions = model_store.predict_many(
            texts=texts,
            model_name=model_name,
            row_indexes=row_indexes,
        )

        return {
            "model_name": model_name,
            "filename": file.filename,
            "detected_text_column": detected_column,
            "count": len(predictions),
            "predictions": predictions,
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"CSV prediction failed: {str(error)}",
        )


def validate_model_name(model_name):
    if model_name not in model_store.get_loaded_model_names():
        raise HTTPException(
            status_code=400,
            detail=f"unknown model_name '{model_name}'",
        )


def validate_csv_file(file):
    filename = file.filename or ""

    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="uploaded file must be a CSV file",
        )


def detect_text_column(df, requested_column):
    if requested_column:
        if requested_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"requested text_column '{requested_column}' not found in CSV",
            )

        return requested_column

    lower_to_original = {
        column.lower(): column
        for column in df.columns
    }

    for candidate in CSV_TEXT_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    raise HTTPException(
        status_code=400,
        detail=(
            "could not detect text column. "
            f"Use one of these column names: {CSV_TEXT_COLUMN_CANDIDATES}, "
            "or pass text_column in the form data."
        ),
    )