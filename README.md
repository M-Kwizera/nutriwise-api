# NutriWise API

FastAPI service for the NutriWise machine-learning project. It loads the fitted Task 1 pipeline from `best_model.joblib` and exposes prediction, health-check, and retraining endpoints.

## What the prediction means

The model predicts `DR1TKCAL`: Day 1 reported dietary energy intake (kcal) from one NHANES 24-hour dietary recall. It is **not** a recommended calorie target, daily energy expenditure, or medical advice. The training data is from the United States NHANES 2021–2023 cycle; validate on representative Rwandan data before using it for local decisions.

## Files

- `main.py` — FastAPI application
- `best_model.joblib` — trained preprocessing + SGDRegressor pipeline from Task 1
- `nutriwise_dataset.csv` — source data used for non-incremental retraining fallback
- `requirements.txt` — Python dependencies
- `render.yaml` — Render deployment configuration
- `sample_request.json` — example prediction request

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger documentation.

## Endpoints

- `GET /` — service information
- `GET /health` — health check
- `POST /predict` — JSON input with all 10 model features
- `POST /retrain` — multipart upload named `file`, containing a CSV with the 10 features plus `DR1TKCAL`

The uploaded retraining CSV must contain at least 10 rows with valid, non-negative `DR1TKCAL` values. The current SGDRegressor supports incremental learning using `partial_fit`.

## Deploy to Render

This repository is structured with `main.py` and `requirements.txt` at its root, so leave Render's **Root Directory** blank.

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

After deployment, open your Render service URL and append `/docs` to test the interactive API documentation.

Set `ALLOWED_ORIGINS` to a comma-separated list of exact frontend origins if using Flutter Web. Native Android/iOS apps do not use browser CORS enforcement. The Render filesystem may be ephemeral on a free service, so a retrained model can revert after a restart or redeployment.
