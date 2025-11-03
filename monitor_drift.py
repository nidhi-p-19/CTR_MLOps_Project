# monitor_drift.py
"""
Automated Weekly Drift Monitoring
---------------------------------
This script:
1️⃣ Loads reference & current data (from /data or DVC)
2️⃣ Runs Evidently’s DataDriftPreset
3️⃣ Generates a full HTML drift report
4️⃣ Saves it to /reports/drift_report.html
5️⃣ Prints drift summary to console (for GitHub Actions log)
"""

import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# --------- Ensure paths ---------
os.makedirs("reports", exist_ok=True)
data_path = "data/50krecords.csv"
current_path = "data/current_batch.csv"  # simulate new data batch

# --------- Load Data ---------
if not os.path.exists(data_path):
    raise FileNotFoundError("Reference dataset not found. Run `dvc pull` first.")

reference = pd.read_csv(data_path)

# Simulate 'current' data for drift detection
# (In production, you'd replace this with new data from your pipeline)
if not os.path.exists(current_path):
    current = reference.sample(frac=0.8, random_state=42)
    current.to_csv(current_path, index=False)
else:
    current = pd.read_csv(current_path)

# --------- Run Evidently Report ---------
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)

# --------- Save HTML Report ---------
report.save_html("reports/drift_report.html")

# --------- Print Summary ---------
drift_summary = report.as_dict()
n_features = drift_summary["metrics"][0]["result"]["number_of_columns"]
drifted = drift_summary["metrics"][0]["result"]["number_of_drifted_columns"]
share = drift_summary["metrics"][0]["result"]["share_of_drifted_columns"]

print("\n🧠 Drift Monitoring Summary:")
print(f"  Total features analyzed: {n_features}")
print(f"  Drifted features:        {drifted}")
print(f"  Drift ratio:             {share*100:.2f}%")

if drifted > 0:
    print("⚠️  Data drift detected! Check 'reports/drift_report.html' for full details.")
else:
    print("✅ No significant data drift detected.")
