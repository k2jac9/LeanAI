from metaflow import FlowSpec, step, Parameter, current
import pandas as pd
import numpy as np
import joblib
import mlflow
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

class BodyFatPredictionFlow(FlowSpec):

    test_size = Parameter("test_size", default=0.2)
    random_state = Parameter("random_state", default=42)

    @step
    def start(self):
        print("🔁 Loading and cleaning data...")
        df = pd.read_csv("data/kaggle_datasets/body_fat/bodyfat.csv").dropna()

        # Encode 'Sex' column if needed (string to int)
        if df["Sex"].dtype == object:
            df["Sex"] = df["Sex"].map({"M": 1, "F": 0})

        # Manual feature engineering
        df["bmi"] = df["Weight"] / (df["Height"] / 100) ** 2
        df["waist_to_hip"] = df["Abdomen"] / df["Hip"]
        df["waist_to_height"] = df["Abdomen"] / df["Height"]
        df["arm_ratio"] = df["Forearm"] / df["Biceps"]

        # Outlier removal
        from scipy.stats import zscore
        z_scores = np.abs(zscore(df.select_dtypes(include=np.number)))
        df = df[(z_scores < 3).all(axis=1)]

        self.df = df
        self.next(self.split_data)

    @step
    def split_data(self):
        df = self.df

        self.df_combined = df.copy()
        self.df_male = df[df["Sex"] == 1].copy()
        self.df_female = df[df["Sex"] == 0].copy()

        print("📊 Combined:", len(self.df_combined), "| Male:", len(self.df_male), "| Female:", len(self.df_female))
        self.next(self.train_models)

    def prepare_data(self, df):
        X = df.drop(columns=["BodyFat", "Original"], errors="ignore")
        y = df["BodyFat"]

        poly = PolynomialFeatures(degree=2, include_bias=False)
        rfe = RFE(LinearRegression(), n_features_to_select=8)
        pca = PCA(n_components=5)

        X_poly = poly.fit_transform(X)
        X_selected = rfe.fit_transform(X_poly, y)
        X_pca = pca.fit_transform(X_selected)

        return train_test_split(X_pca, y, test_size=self.test_size, random_state=self.random_state)

    def log_model(self, X_train, X_test, y_train, y_test, model_name):
        mlflow.set_experiment("BodyFat_Prediction_Metaflow")
        reports_path = Path("reports")
        models_path = Path("models")
        reports_path.mkdir(exist_ok=True)
        models_path.mkdir(exist_ok=True)

        with mlflow.start_run(run_name=model_name):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", SVR(C=10, epsilon=0.01, kernel="rbf"))
            ])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log metrics
            metrics = {
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred),
                "Explained_Variance": explained_variance_score(y_test, y_pred)
            }
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Save and log model
            model_path = models_path / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            mlflow.sklearn.log_model(model, model_name)
            mlflow.log_artifact(str(model_path))

            # Generate and log Evidently reports
            y_df = pd.DataFrame({"target": y_test.reset_index(drop=True), "prediction": y_pred})

            report = Report(metrics=[RegressionPreset()])
            report.run(reference_data=y_df, current_data=y_df)
            report_path = reports_path / f"{model_name}_report.html"
            report.save_html(str(report_path))
            mlflow.log_artifact(str(report_path))

    @step
    def train_models(self):
        print("🚀 Training combined model...")
        X_train, X_test, y_train, y_test = self.prepare_data(self.df_combined)
        self.log_model(X_train, X_test, y_train, y_test, "Combined_Model")

        if len(self.df_male) > 10:
            print("🚹 Training male model...")
            X_train_m, X_test_m, y_train_m, y_test_m = self.prepare_data(self.df_male)
            self.log_model(X_train_m, X_test_m, y_train_m, y_test_m, "Male_Model")

        if len(self.df_female) > 10:
            print("🚺 Training female model...")
            X_train_f, X_test_f, y_train_f, y_test_f = self.prepare_data(self.df_female)
            self.log_model(X_train_f, X_test_f, y_train_f, y_test_f, "Female_Model")

        self.next(self.end)

    @step
    def end(self):
        print("✅ Done! All models trained and tracked with MLflow.")

if __name__ == "__main__":
    BodyFatPredictionFlow()
