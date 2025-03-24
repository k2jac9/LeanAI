# LeanAI – Predicting Body Fat Percentage Using Machine Learning

## 1️⃣ Project Overview

This project leverages the **Body Fat Prediction dataset from Kaggle**, which includes anthropometric measurements (e.g., height, weight, waist) to predict **body fat percentage** using machine learning.

---

### 🔹 Business Objective

🎯 Build a machine learning model to estimate **body fat percentage** from physical measurements.

---

### 📌 Why is This Important?

- 🏋️ **Fitness & Health**: Personalize diet and exercise plans.
- 🏥 **Medical Diagnosis**: Assess risks for obesity-related conditions.
- 🏆 **Sports Optimization**: Ideal body composition for athletic performance.
- 💰 **Insurance**: Risk-based premium adjustment.

---

### 💼 Business Use Cases

#### 1. Health & Fitness Industry

- **Motivation**: Replace expensive tools (DEXA) with scalable solutions.
- **Use Case**: Gyms or fitness apps provide instant assessments.
- **Value**: Low-cost health insights boost engagement and subscriptions.

#### 2. SaaS Wellness Tools

- **Use Case**: Scalable B2B solutions for clinics, schools, or corporations.
- **Value**: Enables health monitoring at scale without medical devices.

#### 3. Personalized E-Commerce

- **Use Case**: Tailored plans (meals, workouts, apparel) based on body fat.
- **Value**: Higher customer satisfaction → more conversions and loyalty.

#### 4. Insurance & Public Health

- **Use Case**: More accurate risk profiling than BMI alone.
- **Value**: Better risk management for insurers, better outcomes for health programs.

---

## 2️⃣ Dataset & Features

- 📂 **Source**: Kaggle – Body Fat Prediction
- 💾 **Size**: 436 samples, 16 columns
- 🧪 **Features**:
  - **Target**: BodyFat (percentage)
  - **Inputs**: Age, Weight, Height, Abdomen, Chest, Neck, Thigh, Hip, etc.

---

## 3️⃣ Methodology

### 📊 Exploratory Data Analysis (EDA)

- [Bodyfat EDA Methodology](notebooks/eda/README.md)

- Dataset was **clean**, numeric, and no nulls.
- **Visualizations** revealed normal distributions with minor skewness.
- Strong correlations:
  - **Negative**: BodyFat vs Density
  - **Positive**: Abdomen & Chest vs BodyFat
- **Sex-based analysis** showed anatomical differences, but **Sex** had a weak impact on body fat prediction.

📈 Key Insights:

- **Abdomen circumference** is the strongest single predictor.
- **Weight** correlates more with muscle mass than fat.

---

### 🧠 Feature Engineering

Custom features:

- `bmi = Weight / (Height/100)^2`
- `waist_to_hip = Abdomen / Hip`
- `waist_to_height = Abdomen / Height`
- `arm_ratio = Forearm / Biceps`

---

### 🧪 Modeling Strategy

Started simple due to data size:

- ✅ **Linear Regression** (baseline)
- ✅ **SVR + RFE + PCA** (enhanced model)
- ✅ **Separate models**: Male / Female / Combined

**Metrics**:

- MAE, RMSE, R²
- Evidently AI reports for:
  - Data Drift
  - Target Drift
  - Regression Performance

---

## 4️⃣ Machine Learning Stack

- ⚙️ **Data Processing**: Polars, Scikit-Learn
- 🔁 **Workflow**: Metaflow
- 📊 **Experiment Tracking**: MLflow
- 🔍 **Monitoring**: Evidently AI
- 🧪 **Tuning**: Optuna
- 📦 **API**: FastAPI
- 📤 **Serving**: Streamlit dashboard
- 📂 **Model Storage**: joblib + MLflow Artifacts

---

## 5️⃣ FastAPI Web & REST API

```bash
# Docker Deployment
docker-compose up --build   # First time
docker-compose up           # Regular run

# Local Access
http://localhost:8000       # Web form
http://localhost:8000/docs  # Swagger UI

Sample curl request:
curl -X POST "http://localhost:8000/predict/" \
     -H "accept: application/json" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "abdomen=110&hip=120&weight=100&thigh=190&knee=50&biceps=38&neck=45"
```

---

## 6️⃣ Expected Outcomes

- ✅ **Accurate prediction** of body fat percentage
- ✅ **Feature-based health insights** using anthropometric measurements
- ✅ **Scalable FastAPI** for real-world integration (web & REST)
- ✅ **Visual analytics** using Evidently AI for drift detection and retraining triggers

---

## 🎓 Team Members

| Name                 | Email                          | Video Link |
|----------------------|--------------------------------|------------|
| Igor Bak             | baxwork88@gmail.com            | 283        |
| Alejandro Castellanos| alexcastellanos29@gmail.com    | 283        |
| Faisal Khan          | fa.khan@alumni.utoronto.ca     | 283        |
| Hassan Saade         | saadehassan@hotmail.com        | 283        |
| Anna Wong            | annawong.work@gmail.com        | 283        |
