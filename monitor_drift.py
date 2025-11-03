import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# Ensure reports directory exists
os.makedirs("reports", exist_ok=True)

# Load reference and new data
reference = pd.read_csv("data/50krecords.csv")
current = pd.read_csv("data/50krecords.csv")

# Example: simulate small drift
current = current.sample(frac=1).reset_index(drop=True)

# Define drift report
report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset()
])

# Run and save
report.run(reference_data=reference, current_data=current)
report.save_html("reports/drift_report.html")

print("✅ Drift report generated → reports/drift_report.html")
