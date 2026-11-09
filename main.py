"""NutriWise FastAPI service for estimating NHANES Day 1 reported calorie intake."""
from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "best_model.joblib"))
TRAINING_DATA_PATH = Path(os.getenv("TRAINING_DATA_PATH", BASE_DIR / "nutriwise_dataset.csv"))

NUMERIC_FEATURES = [
    "RIDAGEYR", "DMDHHSIZ", "INDFMPIR", "BMXHT", "BMXWT", "BMXBMI", "BMXWAIST"
]
CATEGORICAL_FEATURES = ["RIAGENDR", "RIDRETH3", "DMDEDUC2"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "DR1TKCAL"

# Categorical codes match the NHANES coding used when training the pipeline.
GenderCode = Literal[1, 2]
RaceEthnicityCode = Literal[1, 2, 3, 4, 6, 7]
EducationCode = Literal[1, 2, 3, 4, 5]


class PredictionInput(BaseModel):
    """One person's input values, using the original NHANES column names."""
    model_config = ConfigDict(extra="forbid")

    RIDAGEYR: float = Field(ge=0, le=120, description="Age in years")
    DMDHHSIZ: float = Field(ge=1, le=20, description="Household size")
    INDFMPIR: float = Field(ge=0, le=5, description="Family income-to-poverty ratio")
    BMXHT: float = Field(ge=50, le=250, description="Standing height in centimetres")
    BMXWT: float = Field(ge=10, le=350, description="Weight in kilograms")
    BMXBMI: float = Field(ge=5, le=100, description="Body mass index")
    BMXWAIST: float = Field(ge=30, le=200, description="Waist circumference in centimetres")
    RIAGENDR: GenderCode = Field(description="NHANES gender code: 1 or 2")
    RIDRETH3: RaceEthnicityCode = Field(description="NHANES race/ethnicity code: 1, 2, 3, 4, 6 or 7")
    DMDEDUC2: EducationCode = Field(description="NHANES education code: 1 to 5")


class PredictionOutput(BaseModel):
    predicted_day1_kcal: float
    unit: str = "kcal"
    interpretation: str
    model: str


app = FastAPI(
    title="NutriWise API",
    description=(
        "Predicts Day 1 reported dietary energy intake (DR1TKCAL) from selected "
        "NHANES demographic and body-measurement features. This is not a calorie "
        "target, medical recommendation, or estimate of energy expenditure."
    ),
    version="1.0.0",
)

# Configure deployed frontend origins with ALLOWED_ORIGINS, comma-separated.
# Example: https://your-flutter-web-app.example,https://your-frontend.example
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:  # fail clearly on startup if deployment files are missing/corrupt
    raise RuntimeError(f"Could not load model at {MODEL_PATH}: {exc}") from exc

training_lock = Lock()


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same categorical representation used during Task 1."""
    frame = frame[FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        # OneHotEncoder was trained on string category codes.
        frame[column] = frame[column].map(
            lambda value: str(int(value)) if pd.notna(value) else np.nan
        )
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        # Match the SAS special-missing cleanup in the notebook.
        frame.loc[frame[column].abs() < 1e-70, column] = np.nan
    return frame


@app.get("/")
def root():
    return {
        "message": "NutriWise API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    values = payload.model_dump()
    row = _prepare_frame(pd.DataFrame([values]))
    try:
        prediction = float(model.predict(row)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    # The estimator is a regression model; clip impossible negative outputs.
    prediction = max(0.0, prediction)
    return PredictionOutput(
        predicted_day1_kcal=round(prediction, 1),
        interpretation=(
            "Estimated Day 1 reported calorie intake from one 24-hour dietary recall. "
            "It is not a recommended daily calorie target."
        ),
        model=type(model.named_steps["model"]).__name__,
    )


@app.post("/retrain")
def retrain(file: UploadFile = File(...)):
    """Continue SGD training with an uploaded CSV containing FEATURES + DR1TKCAL."""
    global model
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")

    try:
        uploaded = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc

    required_columns = FEATURES + [TARGET]
    missing = [column for column in required_columns if column not in uploaded.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"message": "CSV is missing required columns.", "missing_columns": missing},
        )

    clean = uploaded[required_columns].copy()
    clean[TARGET] = pd.to_numeric(clean[TARGET], errors="coerce")
    clean = clean.dropna(subset=[TARGET])
    clean = clean[clean[TARGET] >= 0]
    if len(clean) < 10:
        raise HTTPException(
            status_code=422,
            detail="CSV must contain at least 10 rows with valid, non-negative DR1TKCAL values.",
        )

    x_new = _prepare_frame(clean)
    y_new = clean[TARGET].astype(float)

    with training_lock:
        try:
            candidate = joblib.load(MODEL_PATH)
            estimator = candidate.named_steps["model"]
            if hasattr(estimator, "partial_fit"):
                transformed = candidate.named_steps["preprocessor"].transform(x_new)
                estimator.partial_fit(transformed, y_new.to_numpy())
            else:
                # For a non-incremental estimator, refit on original + uploaded data.
                original = pd.read_csv(TRAINING_DATA_PATH)
                original = original[required_columns].copy()
                combined = pd.concat([original, clean], ignore_index=True)
                combined[TARGET] = pd.to_numeric(combined[TARGET], errors="coerce")
                combined = combined.dropna(subset=[TARGET])
                candidate.fit(
                    _prepare_frame(combined), combined[TARGET].astype(float)
                )

            temporary_path = MODEL_PATH.with_suffix(".tmp.joblib")
            joblib.dump(candidate, temporary_path)
            temporary_path.replace(MODEL_PATH)
            model = candidate
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}") from exc

    return {
        "message": "Model retrained successfully.",
        "rows_used": int(len(clean)),
        "model": type(model.named_steps["model"]).__name__,
        "note": "The updated model file is saved in the service filesystem.",
    }
