from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import (
    ALLOWED_MODELS,
    DEFAULT_MODEL_NAME,
    MAX_BATCH_SIZE,
    MAX_TEXT_LENGTH,
    MIN_TEXT_LENGTH,
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="tweet or social media text to classify",
        examples=["you are so stupid"],
    )

    model_name: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="model name to use for prediction",
        examples=["best"],
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_blank(cls, value):
        if not value.strip():
            raise ValueError("text cannot be blank")
        return value

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value):
        if value not in ALLOWED_MODELS:
            raise ValueError(f"model_name must be one of {sorted(ALLOWED_MODELS)}")
        return value


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: List[str] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="list of tweets or social media texts to classify",
        examples=[["you are stupid", "hope you have a nice day"]],
    )

    model_name: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="model name to use for prediction",
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values):
        for text in values:
            if not isinstance(text, str):
                raise ValueError("all items must be strings")

            if not text.strip():
                raise ValueError("texts cannot contain blank values")

            if len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:
                raise ValueError(
                    f"each text must be between {MIN_TEXT_LENGTH} and {MAX_TEXT_LENGTH} characters"
                )

        return values

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value):
        if value not in ALLOWED_MODELS:
            raise ValueError(f"model_name must be one of {sorted(ALLOWED_MODELS)}")
        return value


class PredictionResponse(BaseModel):
    row_index: Optional[int] = None
    model_name: str
    input_text: str
    cleaned_text: str
    label_id: int
    label_name: str
    confidence: float
    scores: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    model_name: str
    count: int
    predictions: List[PredictionResponse]


class CsvPredictionResponse(BaseModel):
    model_name: str
    filename: str
    detected_text_column: str
    count: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]


class ModelsResponse(BaseModel):
    available_models: List[str]
    default_model: str