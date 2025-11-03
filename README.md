# CTR_MLOps_Project (Logistic Regression + Free MLOps)

## What this includes
- **train.py**: trains Logistic Regression on `data/train.csv`, evaluates on `data/val.csv`, logs to MLflow, saves model.
- **app.py**: FastAPI service exposing `/predict`.
- **dvc.yaml**: data/model versioning (run `dvc repro`).
- **.github/workflows/mlops.yml**: GitHub Action to retrain on every push.
- **requirements.txt**: minimal deps.

## Quickstart
1) Put your exported Kaggle files here:
```
data/train.csv
data/val.csv
model/logreg_ctr_model.pkl   # optional; will be overwritten by train.py
```

2) Create venv & install deps
```
pip install -r requirements.txt
```

3) (Optional) start MLflow UI in a new terminal
```
mlflow ui
```

4) Train
```
python train.py
```

5) Serve the model
```
uvicorn app:app --reload
# open http://127.0.0.1:8000/docs
```

6) Enable DVC (optional but recommended)
```
dvc init
dvc repro
```

7) CI/CD
- Push this repo to GitHub (branch `main`). GitHub Actions will automatically run `python train.py`.

## Predict payload example
```json
{
  "C1": 2,
  "banner_pos": 0,
  "site_id": 582,
  "site_domain": 7339,
  "site_category": 2,
  "app_id": 7884,
  "app_domain": 254,
  "app_category": 0,
  "device_id": 123,
  "device_ip": 456,
  "device_model": 789,
  "device_type": 1,
  "device_conn_type": 2,
  "C14": 123,
  "C15": 320,
  "C16": 50,
  "C17": 157,
  "C18": 0,
  "C19": 32,
  "C20": 65,
  "C21": 52,
  "ad_popularity": 0.0,
  "user_past_ctr": 0.0,
  "session_len": 1.0,
  "hour_of_day": 12,
  "day_of_week": 3,
  "is_weekend": 0
}
```

> Ensure the feature names match your preprocessed columns (excluding `click`, `id`, `hour`, `datetime`).

