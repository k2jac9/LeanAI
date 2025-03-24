import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset
import joblib
from scipy.stats import zscore

# Load dataset
df = pd.read_csv("data/kaggle_datasets/body_fat/bodyfat.csv")
df.dropna(inplace=True)

# Normalize 'Sex' to binary
df["Sex"] = df["Sex"].map({"M": 1, "F": 0})

# Manual Feature Engineering (based on YAMLs)
df["bmi"] = df["Weight"] / (df["Height"] / 100) ** 2
df["waist_to_hip"] = df["Abdomen"] / df["Hip"]
df["waist_to_height"] = df["Abdomen"] / df["Height"]
df["arm_ratio"] = df["Forearm"] / df["Biceps"]

# Remove outliers using z-score
z_scores = np.abs(zscore(df.select_dtypes(include=np.number)))
df = df[(z_scores < 3).all(axis=1)]

# Paths
reports_path = Path("reports")
models_path = Path("models")
reports_path.mkdir(exist_ok=True)
models_path.mkdir(exist_ok=True)

# Define features and target
X_raw = df.drop(columns=["BodyFat", "Original"])
y = df["BodyFat"]

# Feature engineering pipeline
poly = PolynomialFeatures(degree=2, include_bias=False)
rfe = RFE(LinearRegression(), n_features_to_select=8)
pca = PCA(n_components=5)

# Encode categorical column
if "Sex" in X_raw.columns and X_raw["Sex"].dtype == object:
    X_raw["Sex"] = X_raw["Sex"].map({"M": 1, "F": 0})

# Transform full dataset
X_poly = poly.fit_transform(X_raw)
X_selected = rfe.fit_transform(X_poly, y)
X_pca = pca.fit_transform(X_selected)
X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])

X_train, X_test, y_train, y_test = train_test_split(X_pca_df, y, test_size=0.2, random_state=42)

# MLflow setup
mlflow.set_experiment("BodyFat_Prediction_Advanced")

def log_metrics(name, y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "Explained_Variance": explained_variance_score(y_true, y_pred)
    }

def generate_evidently_reports(X_train, X_test, y_true, y_pred, prefix):
    # Rebuild dataframes with consistent column names
    drift_reference = pd.DataFrame(X_train)
    drift_current = pd.DataFrame(X_test)

    target_drift_data = pd.DataFrame({
        "target": y_true.reset_index(drop=True),
        "prediction": y_pred
    })

    # Data Drift
    drift = Report(metrics=[DataDriftPreset()])
    drift.run(reference_data=drift_reference, current_data=drift_current)
    drift_path = reports_path / f"{prefix}_drift_report.html"
    drift.save_html(str(drift_path))

    # Target Drift
    target_drift = Report(metrics=[TargetDriftPreset()])
    target_drift.run(reference_data=target_drift_data, current_data=target_drift_data)
    target_drift_path = reports_path / f"{prefix}_target_drift.html"
    target_drift.save_html(str(target_drift_path))

    # Regression Performance
    reg_report = Report(metrics=[RegressionPreset()])
    reg_report.run(reference_data=target_drift_data, current_data=target_drift_data)
    reg_perf_path = reports_path / f"{prefix}_regression_perf.html"
    reg_report.save_html(str(reg_perf_path))

    return [drift_path, target_drift_path, reg_perf_path]


def train_and_log_model(X_train, X_test, y_train, y_test, model_name, save_path):
    with mlflow.start_run(run_name=model_name):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", SVR(C=10, epsilon=0.01, kernel="rbf"))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = log_metrics(model_name, y_test, y_pred)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, model_name)

        reports = generate_evidently_reports(X_train, X_test, y_test, y_pred, prefix=model_name)
        for r in reports:
            mlflow.log_artifact(str(r))

        model_path = models_path / f"{save_path}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

# 🔥 Train Combined Model
train_and_log_model(X_train, X_test, y_train, y_test, "Combined_Model", "combined_model")

# 🔥 Train Male Model
df_male = df[df["Sex"] == 1]
X_m_raw = df_male.drop(columns=["BodyFat", "Original"])
y_m = df_male["BodyFat"]
X_m_poly = poly.fit_transform(X_m_raw)
X_m_selected = rfe.fit_transform(X_m_poly, y_m)
X_m_pca = pca.fit_transform(X_m_selected)
X_m_pca_df = pd.DataFrame(X_m_pca, columns=[f"PC{i+1}" for i in range(X_m_pca.shape[1])])
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_m_pca_df, y_m, test_size=0.2, random_state=42)
train_and_log_model(X_train_m, X_test_m, y_train_m, y_test_m, "Male_Model", "male_model")

# 🔥 Train Female Model
df_female = df[df["Sex"] == 0]
X_f_raw = df_female.drop(columns=["BodyFat", "Original"])
y_f = df_female["BodyFat"]
X_f_poly = poly.fit_transform(X_f_raw)
X_f_selected = rfe.fit_transform(X_f_poly, y_f)
X_f_pca = pca.fit_transform(X_f_selected)
X_f_pca_df = pd.DataFrame(X_f_pca, columns=[f"PC{i+1}" for i in range(X_f_pca.shape[1])])
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_f_pca_df, y_f, test_size=0.2, random_state=42)
train_and_log_model(X_train_f, X_test_f, y_train_f, y_test_f, "Female_Model", "female_model")
