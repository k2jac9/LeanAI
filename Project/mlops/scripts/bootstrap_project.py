import os

BASE_DIR = "C:/Users/AC/Projects/LeanAI/Project/mlops"

structure = {
    "features": {
        "bmi.yaml": """\
entity:
  name: individual
  primary_key: id

feature_view:
  name: bmi_feature
  description: BMI feature from Weight and Height
  entities:
    - individual
  features:
    - name: bmi
      type: float
      transformation: |
        bmi = Weight / (Height / 100)**2
""",
        "waist_to_hip.yaml": """\
entity:
  name: individual
  primary_key: id

feature_view:
  name: waist_to_hip_ratio
  description: Waist to hip ratio
  entities:
    - individual
  features:
    - name: waist_to_hip
      type: float
      transformation: |
        waist_to_hip = Abdomen / Hip
""",
        "waist_to_height.yaml": """\
entity:
  name: individual
  primary_key: id

feature_view:
  name: waist_to_height_ratio
  description: Waist to height ratio
  entities:
    - individual
  features:
    - name: waist_to_height
      type: float
      transformation: |
        waist_to_height = Abdomen / Height
""",
        "forearm_to_biceps.yaml": """\
entity:
  name: individual
  primary_key: id

feature_view:
  name: forearm_to_biceps_ratio
  description: Ratio of Forearm to Biceps
  entities:
    - individual
  features:
    - name: forearm_to_biceps
      type: float
      transformation: |
        forearm_to_biceps = Forearm / Biceps
"""
    },
    "reports": {
        "data_drift_template.html": "<html><head><title>Data Drift Report</title></head><body><h1>Placeholder for Drift Report</h1></body></html>",
        "target_drift_template.html": "<html><head><title>Target Drift Report</title></head><body><h1>Placeholder for Target Drift Report</h1></body></html>",
        "regression_performance_template.html": "<html><head><title>Regression Performance</title></head><body><h1>Placeholder for Performance Metrics</h1></body></html>"
    },
    "tests": {
        "test_data_loading.py": """\
import pandas as pd

def test_load_data():
    df = pd.read_csv('data/kaggle_datasets/body_fat/bodyfat.csv')
    assert not df.empty, "Loaded DataFrame is empty"
    assert 'BodyFat' in df.columns, "'BodyFat' column is missing"
"""
    },
    "notebooks": {
        "01_eda.ipynb": "",  # Blank starter notebook
    },
    ".github/workflows": {
        "mlops_pipeline.yaml": """\
name: MLOps Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.12

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Run tests
      run: pytest tests/
"""
    },
    "scripts": {
        "train_and_log_mlflow.py": """\
# This is a stub for the training + MLflow logging script
print("Training pipeline entry point. Fill in your logic here.")
"""
    }
}

for folder, files in structure.items():
    folder_path = os.path.join(BASE_DIR, folder)
    os.makedirs(folder_path, exist_ok=True)
    for filename, content in files.items():
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

print("✅ Project bootstrapped with features, reports, scripts, CI/CD, and tests!")