# train.py — CTR Ad Recommendation Model (LightGBM + DVC + MLflow)
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
import lightgbm as lgb
import yaml

# ------------------- CONFIG -------------------
DATA_PATH = "data/50krecords.csv"
MODEL_PATH = "model/ctr_model_lgb.pkl"
FEATS_PATH = "model/feature_names.pkl"
EXPERIMENT_NAME = "CTR_Ad_Recommendation_LightGBM"

# ------------------- MAIN ---------------------
def main():
    print("⚙️ Training LightGBM model...")

    # 1️⃣ Load parameters
    with open("params.yaml", "r") as f:
        params_all = yaml.safe_load(f)
    lgb_params = params_all["lightgbm"]

    # Remove unsupported params
    sample_size = lgb_params.pop("sample_size", None)

    # 2️⃣ Load dataset
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    # 3️⃣ Convert categorical columns to numeric
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    y = df["click"]
    X = df.drop(columns=["click"])

    # 4️⃣ Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=lgb_params["seed"]
    )

    # 5️⃣ Initialize MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_params(lgb_params)
        if sample_size:
            mlflow.log_param("sample_size", sample_size)

        # 6️⃣ Train model
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(-1)]  # disables training log spam
        )

        # 7️⃣ Predict and evaluate
        p = model.predict_proba(X_val)[:, 1]
        p = np.clip(p, 0, 1)

        auc = roc_auc_score(y_val, p)
        logloss = log_loss(y_val, p)
        pr_auc = average_precision_score(y_val, p)

        # 8️⃣ Log metrics
        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("LogLoss", logloss)
        mlflow.log_metric("PR_AUC", pr_auc)

        # 9️⃣ Save model + schema
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(list(X.columns), FEATS_PATH)

        mlflow.lightgbm.log_model(model, "model")

        # 🔟 Save metrics for DVC
        metrics = {"AUC": auc, "LogLoss": logloss, "PR_AUC": pr_auc}
        os.makedirs("metrics", exist_ok=True)
        import json
        with open("metrics/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"✅ Saved model + schema. AUC={auc:.4f} LogLoss={logloss:.4f} PR-AUC={pr_auc:.4f}")


if __name__ == "__main__":
    main()
