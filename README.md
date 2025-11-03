# 🚀 CTR_MLOps_Project  
**End-to-End Click-Through Rate (CTR) Prediction Pipeline**  
*Powered by LightGBM, DVC, MLflow, and FastAPI*

---

## 🧠 Overview
This project implements a full **MLOps workflow** for predicting **ad click-through rates (CTR)**.  
It integrates:
- **ML model training** (LightGBM)
- **Data & model versioning** (DVC)
- **Experiment tracking** (MLflow)
- **API deployment** (FastAPI)

All stages are automated and reproducible — from dataset preprocessing to serving predictions via REST API.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[Raw Data (CSV)] -->|Tracked with DVC| B[Training Pipeline (train.py)]
    B --> C[LightGBM Model]
    C --> D[Model Registry (MLflow)]
    D --> E[FastAPI App (app.py)]
    E --> F[Prediction Endpoint /predict]
    B --> G[Metrics & Params (metrics.json / params.yaml)]
```

---

## ⚙️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Version Control** | Git + GitHub |
| **Data/Model Versioning** | DVC |
| **Experiment Tracking** | MLflow |
| **Modeling** | LightGBM |
| **Deployment** | FastAPI + Uvicorn |
| **Dependency Management** | requirements.txt |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
CTR_MLOps_Project/
│
├── data/                # Dataset tracked via DVC
│   ├── 50krecords.csv
│   └── 50krecords.csv.dvc
│
├── model/               # Trained LightGBM model
│   ├── ctr_model_lgb.pkl
│   └── feature_names.pkl
│
├── metrics/             # Model performance metrics
│   └── metrics.json
│
├── mlruns/              # MLflow experiments and runs
│
├── app.py               # FastAPI inference API
├── train.py             # Model training + logging script
├── params.yaml          # Model hyperparameters
├── dvc.yaml             # DVC pipeline stages
├── requirements.txt     # Python dependencies
├── .gitignore           # Ignore venv/cache
├── .dvcignore           # Ignore local DVC cache
└── README.md
```

---

## 🔄 Workflow

### 1️⃣ Data Versioning with DVC
```bash
dvc add data/50krecords.csv
git add data/50krecords.csv.dvc data/.gitignore
git commit -m "Track dataset with DVC"
```

### 2️⃣ Train Model
```bash
python train.py
```

- Logs metrics via `metrics/metrics.json`
- Saves model artifacts in `/model`
- Tracks run automatically in MLflow

### 3️⃣ Reproduce Pipeline
```bash
dvc repro
```

This command retrains and regenerates everything based on `params.yaml`.

### 4️⃣ Serve Model via FastAPI
```bash
uvicorn app:app --reload
```
Navigate to:  
➡️ [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Example Input:
```json
{
  "ad_popularity": 0.64,
  "user_past_ctr": 0.38,
  "day_of_week": 3,
  "C18": 2,
  "C19": 7,
  "C20": 4,
  "C14": 215,
  "app_category": 3,
  "site_category": 1
}
```

Example Output:
```json
{
  "click_probability": 0.0388750775927592
}
```

---

## 📊 Example Metrics

| Metric | Value |
|---------|--------|
| **AUC** | 0.7152 |
| **LogLoss** | 0.4147 |
| **PR-AUC** | 0.3240 |

*(logged automatically to MLflow and stored via DVC)*

---

## 📦 Reproducibility

All parameters are controlled from `params.yaml`.  
Modify hyperparameters and rerun with:
```bash
dvc repro
```

Every run automatically generates:
- New model version (`/model`)
- Updated metrics (`metrics.json`)
- MLflow experiment entry

---

## 🧩 Deployment Notes

- You can containerize this with Docker or deploy to Render/Railway easily.
- The FastAPI app is lightweight and ready for production testing.
- For cloud DVC remote setup, link with S3, GDrive, or Azure Blob.

---

## 🧑‍💻 Author
**Nidhi Parmar (nidhi-p-19)**  
CTR Prediction | MLOps | Data Science | FastAPI  
📍 [GitHub Profile](https://github.com/nidhi-p-19)

---

## ⭐ Future Enhancements
- CI/CD via GitHub Actions  
- Model registry sync between MLflow and DVC  
- Streamlit dashboard for real-time metrics  
- Docker-based deployment pipeline  

---

## 🏁 Summary
This project demonstrates a **production-grade MLOps workflow** — integrating **data versioning, model tracking, automation, and deployment** into one reproducible system.

> “Train. Track. Version. Deploy. Repeat.” ⚡
