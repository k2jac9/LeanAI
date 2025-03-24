import os
import shutil
import zipfile
import subprocess
import kagglehub

def run_command(command):
    """Runs a shell command and exits if an error occurs."""
    print(f"🔹 Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Error executing: {command}")
        exit(1)

def create_folders():
    """Ensures the full project structure is created correctly."""
    print("📂 Verifying and creating missing project folders...")

    folders = [
        "infra", "data/raw", "data/processed", "notebooks",
        "src/fastapi_app", "src/retraining", "src/monitoring",
        "models/mlflow", "dashboards", "scripts", "tests", ".github/workflows"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("✅ Folder structure verified!")

    # Ensuring necessary files exist
    files = {
        "infra/local_setup.tf": "",
        "infra/cloud_setup.tf": "",
        "infra/docker-compose.yml": "",
        "notebooks/eda.ipynb": "",
        "notebooks/model-training.ipynb": "",
        "src/fastapi_app/main.py": "# FastAPI Entry Point",
        "src/fastapi_app/logging.py": "# Logging Configuration",
        "src/retraining/retrain_flow.py": "# Metaflow Retraining Script",
        "src/monitoring/drift_detection.py": "# Evidently AI Drift Detection",
        "dashboards/eda_dashboard.py": "# Streamlit EDA Dashboard",
        "dashboards/prediction_dashboard.py": "# Streamlit Prediction UI",
        "scripts/run_fastapi.sh": "#!/bin/bash\nuvicorn src.fastapi_app.main:app --host 0.0.0.0 --port 8000 --reload",
        "scripts/run_mlflow.sh": "#!/bin/bash\nmlflow ui",
        "tests/test_api.py": "# API Unit Tests",
        "tests/test_model.py": "# Model Evaluation Tests",
        ".github/workflows/deploy.yml": "# GitHub Actions CI/CD Setup",
        "requirements.txt": "",
        ".gitignore": "",
    }

    for file, content in files.items():
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(content)

    print("✅ Essential files verified and created if missing!")

    # Placeholder for README.md
    readme_placeholder = """\
# 🏗 MLOps Project ⚙️ Body Fat Prediction 🧑‍🔬

![MLflow Version](https://img.shields.io/badge/MLflow-2.8-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 📌 Table of Contents

- [Overview](#📌-overview)
- [Technologies Used](#🚀-technologies-used)
- [System Architecture](#🏛-system-architecture)
- [Project Folder Structure](#📂-project-folder-structure)
- [Installation & Setup](#📥-installation--setup)
- [Example API Request](#🔥-example-api-request)
- [Troubleshooting](#🚑-troubleshooting)
- [Demo](#🎥-demo---fastapi-in-action)
- [Next Steps](#🚀-next-steps)
- [Author](#🧑‍💻-about-the-author)
- [License](#📜-license)


---

## 📌 Overview

This project implements a full **MLOps pipeline** for **predicting body fat percentage** using various body measurements. It includes:

- **Data processing**
- **Model training & deployment**
- **Monitoring & automated retraining**
- **Alerts & notifications**

---

## 🚀 Technologies Used

| **Component**              | **Tool**                |
| -------------------------- | ----------------------- |
| **Infrastructure**         | OpenTofu                |
| **Workflow Orchestration** | Metaflow                |
| **Data Processing**        | Polars                  |
| **Experiment Tracking**    | MLflow                  |
| **Hyperparameter Tuning**  | Optuna                  |
| **Feature Store**          | Featureform             |
| **Model API**              | FastAPI                 |
| **Model Deployment**       | MLflow Models           |
| **Model Monitoring**       | Evidently AI + MLflow   |
| **Real-time Logging**      | FastAPI Logging         |
| **Retraining**             | Metaflow + Evidently AI |
| **Alerts & Notifications** | Slack / Discord         |
| **Web UI**                 | Streamlit               |
| **Database for Logs**      | PostgreSQL              |

---

## 🏛 System Architecture

```mermaid
flowchart LR
  A["Raw Data (CSV)"] -->|Processed by Polars| B["Feature Engineering"]
  B -->|Stored in Featureform| C["Feature Store"]
  B -->|Train Model| D["MLflow Model Registry"]
  D -->|Deployed to API| E["FastAPI Model API"]
  E -->|Real-time Predictions| F["Streamlit UI"]
  F -->|User Interaction| G["Interactive Dashboard"]
  E -->|Monitors Drift| H["Evidently AI"]
  H -->|Triggers Retraining| I["Metaflow Retraining Pipeline"]
  I -->|Updates Model| D
```

---

## 📂 Project Folder Structure\

```bash
mlops-project/
│── infra/                   # 🏗 Infrastructure (OpenTofu, Docker)
│   ├── local_setup.tf       # OpenTofu script for local deployment
│   ├── cloud_setup.tf       # OpenTofu script for AWS/GCP deployment
│   ├── docker-compose.yml   # Alternative Docker setup
│── data/                    # 📂 Data files
│   ├── raw/                 # Raw datasets (bodyfat-extended.csv)
│   ├── processed/           # Processed datasets
│── notebooks/               # 📓 Jupyter Notebooks
│   ├── eda.ipynb            # Exploratory Data Analysis
│   ├── model-training.ipynb # Training and hyperparameter tuning
│── src/                     # 📜 Source code for the project
│   ├── fastapi_app/         # API code (FastAPI)
│   │   ├── main.py          # FastAPI app for model serving
│   │   ├── logging.py       # Real-time request logging
│   ├── retraining/          # Retraining workflow (Metaflow)
│   │   ├── retrain_flow.py  # Automated retraining script
│   ├── monitoring/          # Monitoring scripts (Evidently AI)
│   │   ├── drift_detection.py  # Detect model drift
│── models/                  # 🏗 Trained models
│   ├── mlflow/              # MLflow model registry
│── dashboards/              # 📊 Streamlit Dashboards
│   ├── eda_dashboard.py     # Dashboard for EDA & insights
│   ├── prediction_dashboard.py # Dashboard for model predictions
│── scripts/                 # 🏗 Helper scripts
│   ├── run_fastapi.sh       # Start FastAPI
│   ├── run_mlflow.sh        # Start MLflow tracking
│── tests/                   # ✅ Unit & Integration tests
│   ├── test_api.py          # API testing
│   ├── test_model.py        # Model evaluation tests
│── .github/                 # 🛠 GitHub Actions CI/CD
│   ├── workflows/
│   │   ├── deploy.yml       # CI/CD automation
│── pixi.toml                # 📦 Pixi dependency manager
│── README.md                # 📖 Project documentation
│── requirements.txt         # 📦 Additional dependencies (if needed)
│── .gitignore               # 🚫 Ignore unnecessary files


```

---

## 📥 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/k2jac9/mlops-project.git
cd mlops-project
```

### 2️⃣ Install Dependencies (Pixi)

```bash
pixi install
```

### 3️⃣ Automate Setup & Deployment
We use a Python CLI (setup.py) to handle installation, deployment, and testing.

```python
python setup.py folders     # Create folders & files
python setup.py install     # Install dependencies
python setup.py deploy      # Deploy infrastructure
python setup.py fastapi     # Run FastAPI API
python setup.py mlflow      # Start MLflow UI
python setup.py streamlit   # Start Streamlit dashboards
python setup.py test        # Run all tests
python setup.py validate    # Validate FastAPI health
```

---

## 🔥 Example API Request

Users can test the FastAPI model via cURL:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{"features": [90, 100, 95, 80, 60, 40, 0.5, 0.47, 1.1, 0.9]}'
```

📍 Response Example:

```json
{
  "predicted_body_fat": 22.45,
}
```

---

## 🚑 Troubleshooting

🔹 Issue: `ModuleNotFoundError: No module named 'mlflow'`

Solution:

```bash
pixi install
```

🔹 Issue: `tofu: command not found`

Solution:

```bash
curl -fsSL https://opentofu.org/install.sh | sh
🔹 Issue: FastAPI API is not accessible
```

- Check logs for errors.
- Ensure port 8000 is not blocked.

Solution:

```bash

python setup.py fastapi
```

---

## 🎥 Demo - FastAPI in Action

📌 GIF shows FastAPI running & returning predictions.

---

## 🚀 Next Steps

✅ Deploy the setup to AWS/GCP using OpenTofu

✅ Run performance tests on the FastAPI API

✅ Optimize training pipeline with Spark/Dask

---

## 🧑‍💻 About the Author

Alejandro Castellanos
📍 Data Scientist based in Toronto (EST/GMT-4/UTC−04:00)

I explore AI technology and data science while diving into neuroscience, physics, and philosophy. Also passionate about hiking, sports, and film as an art form.

>"In the dance of quarks and stardust light,
>Humans tread, both wrong and right.
>The quest for truth is a recurring theme."
>
>"Errors made, in history's shade,
>Lessons learned, yet oft betrayed.
>We ponder existence, our choices, our fate."
>
>"Seek understanding, a collective plan.
>For in the blend of code and emotion's plea,
>Lies the essence of what it means to be."

### 🤝 Let's Connect!

🔗 GitHub: k2jac9

🔗 LinkedIn: Alejandro Castellanos

## 📜 License

This project is licensed under the MIT License."""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_placeholder)

    print("✅ README.md placeholder created!")

    # Placeholder for pixi.toml
    pixi_toml_placeholder = """\
[project]
authors = ["Alejandro Castellanos <k2jac9@users.noreply.github.com>"]
channels = ["conda-forge"]
name = "ai_dev_env"
platforms = ["win-64"]
version = "0.1.0"

[dependencies]
adbc-driver-manager = "*"
azure-identity = "*"
boto3 = "*"
connectorx = "*"
cookiecutter = "*"
dask = "*"
deltalake = "*"
duckdb = "*"
evidently = "*"
fancyimpute = "*"
fastapi = "*"
fastparquet = "*"
google-auth = "*"
h5py = "*"
ipykernel = "*"
ipywidgets = "*"
kaggle = "*"
kagglehub = "*"
matplotlib = "*"
metaflow = "*"
missingno = "*"
mlflow = "*"
modin = "*"
nbformat = "*"
numpy = "*"
opencv = "*"
openpyxl = "*"
opentofu = "*"
optuna = "*"
optuna-dashboard = "*"
pandas = "*"
pango = "*"
plotly = "*"
polars = "*"
psycopg2 = "*"
pyarrow = "*"
pybind11 = "*"
pydantic = "*"
pygam = "*"
pyiceberg = "*"
python = "3.12.*"
python-dotenv = "*"
requests = "*"
ruff = "*"
sacred = "*"
scikit-image = "*"
scikit-learn = "*"
seaborn = "*"
shap = "*"
sqlalchemy = "*"
streamlit = "*"
tqdm = "*"
transformers = "*"
uv = "*"
xgboost = "*"
yfinance = "*"

[pypi-dependencies]
featureform = "*"
torch = "*"
torchaudio = "*"
torchvision = "*"

[pypi-options]
index-url = "https://pypi.org/simple"
# extra-index-urls = ["https://download.pytorch.org/whl/cpu"]

[tasks]
install-tensorflow = "uv pip install tensorflow-cpu==2.19"  # Use a stable version of TensorFlow
"""
    
    with open("pixi.toml", "w", encoding="utf-8") as f:
        f.write(pixi_toml_placeholder)

    print("✅ pixi.toml placeholder created!")

# Creating Jupyter notebooks with content
    notebooks = {
        "notebooks/eda.ipynb": [
            ("markdown", "# 📊 Exploratory Data Analysis (EDA) with Polars"),
            ("code", "import polars as pl\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Load the dataset\ndf = pl.read_csv('../data/raw/BodyFat - Extended.csv')\ndf.head()"),
            ("markdown", "## 🔹 Dataset Summary"),
            ("code", "df.describe()"),
            ("markdown", "## 🔹 Missing Values"),
            ("code", "df.null_count()"),
            ("markdown", "## 🔹 Correlation Heatmap"),
            ("code", "import seaborn as sns\nimport matplotlib.pyplot as plt\n\n# Convert to Pandas for visualization\npd_df = df.to_pandas()\nplt.figure(figsize=(10, 6))\nsns.heatmap(pd_df.corr(), annot=True, cmap='coolwarm')\nplt.title('Correlation Heatmap')\nplt.show()"),
        ],
        "notebooks/model-training.ipynb": [
            ("markdown", "# 🤖 Model Training & Hyperparameter Tuning with Polars"),
            ("code", "import polars as pl\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_absolute_error, r2_score\n\n# Load data\ndf = pl.read_csv('../data/raw/BodyFat - Extended.csv')\nX = df.drop(['BodyFat'])\ny = df['BodyFat']\n\n# Convert to Pandas for Scikit-Learn\nX_train, X_test, y_train, y_test = train_test_split(X.to_pandas(), y.to_pandas(), test_size=0.2, random_state=42)\n\n# Train model\nmodel = LinearRegression().fit(X_train, y_train)\n\n# Evaluate\npredictions = model.predict(X_test)\nprint(f'R2 Score: {r2_score(y_test, predictions):.4f}')"),
        ]
    }

    for notebook_path, cells in notebooks.items():
        if not os.path.exists(notebook_path):
            nb = nbformat.v4.new_notebook()
            for cell_type, content in cells:
                if cell_type == "code":
                    nb.cells.append(nbformat.v4.new_code_cell(content))
                else:
                    nb.cells.append(nbformat.v4.new_markdown_cell(content))

            with open(notebook_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            print(f"✅ Notebook created: {notebook_path}")

    print("✅ Jupyter notebooks populated with starter code!")
    
# Ensuring necessary files exist with Polars-based content
    files = {
        "notebooks/eda.ipynb": """{
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# 📊 Exploratory Data Analysis (Using Polars)\\n"]
                },
                {
                    "cell_type": "code",
                    "execution_count": null,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import polars as pl\\n",
                        "import matplotlib.pyplot as plt\\n",
                        "import seaborn as sns\\n\\n",
                        "# Load the dataset\\n",
                        "df = pl.read_csv('../data/raw/BodyFat - Extended.csv')\\n",
                        "df.head()"
                    ]
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }""",
        "notebooks/model-training.ipynb": """{
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# 🤖 Model Training & Hyperparameter Tuning (Using Polars)\\n"]
                },
                {
                    "cell_type": "code",
                    "execution_count": null,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import polars as pl\\n",
                        "from sklearn.model_selection import train_test_split\\n",
                        "from sklearn.linear_model import LinearRegression\\n",
                        "from sklearn.metrics import mean_absolute_error, r2_score\\n\\n",
                        "# Load data\\n",
                        "df = pl.read_csv('../data/raw/BodyFat - Extended.csv')\\n",
                        "X = df.drop(['BodyFat'])\\n",
                        "y = df['BodyFat']\\n",
                        "X_train, X_test, y_train, y_test = train_test_split(X.to_pandas(), y.to_pandas(), test_size=0.2, random_state=42)\\n",
                        "# Train model\\n",
                        "model = LinearRegression().fit(X_train, y_train)\\n",
                        "# Evaluate\\n",
                        "predictions = model.predict(X_test)\\n",
                        "print(f'R2 Score: {r2_score(y_test, predictions):.4f}')"
                    ]
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }""",
        "src/fastapi_app/main.py": """from fastapi import FastAPI
import polars as pl
import joblib

app = FastAPI()

# Load trained model
model = joblib.load('models/mlflow/latest_model.pkl')

@app.get("/")
def home():
    return {"message": "FastAPI is running!"}

@app.post("/predict/")
def predict(features: list):
    prediction = model.predict([features])
    return {"predicted_body_fat": prediction[0]}
""",
        "src/retraining/retrain_flow.py": """from metaflow import FlowSpec, step
import polars as pl
from sklearn.linear_model import LinearRegression
import joblib

class RetrainModelFlow(FlowSpec):
    @step
    def start(self):
        print("Starting retraining process...")
        self.next(self.train)

    @step
    def train(self):
        df = pl.read_csv('data/raw/BodyFat - Extended.csv')
        X, y = df.drop(['BodyFat']).to_pandas(), df['BodyFat'].to_pandas()
        model = LinearRegression().fit(X, y)
        joblib.dump(model, 'models/mlflow/latest_model.pkl')
        print("✅ Model retrained and saved!")
        self.next(self.end)

    @step
    def end(self):
        print("✅ Retraining flow completed.")

if __name__ == '__main__':
    RetrainModelFlow()
""",
        "src/monitoring/drift_detection.py": """from evidently import ColumnDriftAnalyzer
import polars as pl

def check_drift():
    df = pl.read_csv('data/raw/BodyFat - Extended.csv')
    reference = df.sample(100, seed=42)
    current = df.sample(100, seed=99)
    drift = ColumnDriftAnalyzer()
    drift_results = drift.calculate(reference.to_pandas(), current.to_pandas())
    print("Drift detected:", drift_results.get_results())

if __name__ == "__main__":
    check_drift()
""",
        "dashboards/eda_dashboard.py": """import streamlit as st
import polars as pl

st.title("📊 Exploratory Data Analysis Dashboard (Using Polars)")

df = pl.read_csv("data/raw/BodyFat - Extended.csv")
st.write("### Dataset Overview", df.head())

st.write("### Column Statistics", df.describe())
""",
        "dashboards/prediction_dashboard.py": """import streamlit as st
import requests

st.title("🤖 Body Fat Prediction")

features = st.text_input("Enter features (comma-separated)")
if st.button("Predict"):
    response = requests.post("http://localhost:8000/predict/", json={"features": [float(x) for x in features.split(",")]})
    st.write("### Prediction Result:", response.json())
"""
    }

    for file, content in files.items():
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(content)

    print("✅ Essential files populated with Polars-based scripts!")

def fetch_kaggle_data():
    """Fetches the dataset from Kaggle and moves it to data/raw/. Avoids re-downloading if it already exists."""
    dataset_path = "data/raw"
    dataset_filename = "BodyFat - Extended.csv"
    dataset_file_path = os.path.join(dataset_path, dataset_filename)

    # Check if dataset already exists
    if os.path.exists(dataset_file_path):
        print(f"✅ Dataset already exists: {dataset_file_path}")
        print("⏭ Skipping download.")
        return

    print("📥 Downloading dataset from Kaggle...")
    os.makedirs(dataset_path, exist_ok=True)

    try:
        # Download dataset using kagglehub (default location: .cache)
        path = kagglehub.dataset_download("simonezappatini/body-fat-extended-dataset")
        print(f"✅ Dataset downloaded to: {path}")

        # Ensure the download path exists
        if not os.path.exists(path):
            print(f"❌ Error: Download directory {path} does not exist.")
            return

        # List downloaded files
        downloaded_files = os.listdir(path)
        print(f"📂 Downloaded files: {downloaded_files}")

        # Check if a ZIP file is present
        zip_file = None
        for file in downloaded_files:
            if file.lower().endswith(".zip"):
                zip_file = file
                break  # Stop at the first ZIP file found

        # If a ZIP file is found, extract it
        if zip_file:
            zip_path = os.path.join(path, zip_file)
            extract_path = os.path.join(path, "extracted")

            print(f"📦 Extracting {zip_file}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            # Update path to extracted directory
            path = extract_path
            downloaded_files = os.listdir(path)
            print(f"✅ Extracted files: {downloaded_files}")

        # Find the correct CSV file (case-insensitive search)
        csv_file = None
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(".csv"):
                    csv_file = os.path.join(root, file)
                    break  # Stop at the first CSV file found

        if csv_file:
            shutil.move(csv_file, dataset_file_path)
            print(f"✅ Dataset moved to {dataset_file_path}")
        else:
            print(f"❌ Error: No CSV file found after extraction.")
    
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")

# Mapping actions to functions
actions = {
    "Create project folders": create_folders,
    "Fetch Kaggle dataset": fetch_kaggle_data,
    "Install dependencies": lambda: run_command("pixi install"),
    "Deploy infrastructure": lambda: run_command("tofu init && tofu apply -auto-approve"),
    "Start FastAPI API": lambda: run_command("uvicorn src.fastapi_app.main:app --host 0.0.0.0 --port 8000 --reload"),
    "Start MLflow UI": lambda: run_command("mlflow ui"),
    "Run Streamlit dashboards": lambda: run_command("streamlit run dashboards/eda_dashboard.py & streamlit run dashboards/prediction_dashboard.py"),
    "Run tests": lambda: run_command("pytest tests/"),
    "Validate API health": lambda: run_command("curl -X 'GET' 'http://localhost:8000/healthcheck/' -H 'accept: application/json'"),
}

if __name__ == "__main__":
    while True:
        print("\n🛠 Choose an action:\n")
        
        # Display menu with numbers
        for i, action in enumerate(actions.keys(), start=1):
            print(f"{i}. {action}")

        # Get user input
        try:
            choice = int(input("\nEnter the number of your choice (or 0 to exit): ").strip())
            if choice == 0:
                print("👋 Exiting setup. Goodbye!")
                break  # Exit loop
            elif 1 <= choice <= len(actions):
                selected_action = list(actions.keys())[choice - 1]
                print(f"\n✅ Running: {selected_action}\n")
                actions[selected_action]()  # Execute the selected function
            else:
                print("❌ Invalid choice. Please enter a valid number.")

        except ValueError:
            print("❌ Please enter a valid number.")
