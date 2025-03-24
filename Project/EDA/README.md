# Bodyfat EDA Process

This document describes the end-to-end process followed during exploratory data analysis (EDA) for the **Bodyfat Dataset**. It includes data cleaning, feature engineering, clustering, visualization, and interactive analysis.

---

## 🥉 1. Setup and Imports

Essential libraries are imported for:

- **Data handling**: `pandas`, `polars`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Machine learning**: `scikit-learn`, `umap-learn`
- **System utilities**: `os`, `warnings`, `pathlib`

Environment settings like `LOKY_MAX_CPU_COUNT` are configured and common warnings are suppressed to keep logs clean.

---

## 📅 2. Data Loading

- The dataset is loaded using **Polars** for speed and converted to **Pandas** for compatibility.
- The file path is kept **relative** for flexibility across environments.

---

## 🦜 3. Preprocessing and Cleaning

### Height and Weight Sanity Check

- Suspected inversions (`Height < 1.0`, `Weight > 100`) are **swapped**.

### Filtering Implausible Data

- Records with **extremely low height (< 0.5m)** or **weight (< 10kg)** are removed.

### Missing Values

- All rows with missing values are dropped to ensure clean inputs for analysis and modeling.

---

## ⚙️ 4. Feature Engineering

Four key derived features are created:

- **BMI** = `Weight / (Height^2)`
- **Forearm to Biceps Ratio** = `Forearm / Biceps`
- **Waist to Height Ratio** = `Abdomen / Height`
- **Waist to Hip Ratio** = `Abdomen / Hip`

Additionally:

- **BMI outliers** (> 100) are identified and dropped to prevent distortion.

---

## 📊 5. Exploratory Data Analysis (EDA)

### Correlation and Distribution Checks:

- **Pairplots**: To observe linear/nonlinear relationships.
- **Heatmaps**: Feature-wise correlation matrices.
- **Histograms with KDE**: Including Mean, Mode, and Outlier indicators.

---

## 🚀 6. Clustering Analysis

### Scaling

- All numeric features are standardized using `StandardScaler`.

### Clustering Methods

- **K-Means** (k=3): Partitioning into 3 clusters.
- **DBSCAN**: Density-based clustering.

### Evaluation

- **Silhouette Scores** are calculated to evaluate clustering performance.

### Dimensionality Reduction

- **t-SNE** and **UMAP** for visualizing high-dimensional data in 2D.
- **Hierarchical Clustering** using Ward linkage for dendrogram visualization.

---

## 📉 7. Cluster Interpretations

- **Scatter Plots**: BodyFat vs BMI, colored by clusters.
- **Box Plots**: BMI, Abdomen, and Weight across clusters.

---

## 🤝 8. Demographic Analysis (Age & Sex)

### Age Grouping

- Age is binned into: 18–30, 31–45, 46–60, and 60+.

### Visualization

- **Violin Plots**: BodyFat distribution by AgeGroup and Sex.
- **Boxplots**: Key features across AgeGroups and Sex.
- **Sex-specific Correlation Heatmaps** for top features.

---

## 🌐 9. Interactive Dashboard

- Built using **Plotly Express**.
- An interactive scatter plot: BMI vs BodyFat colored by clusters with hover data.

---

## 🔎 10. Outlier Flagging

- Each numeric column is scanned for outliers using the rule: `|x - mean| > 2 * std`.
- Binary flags are added for each outlier column.

---

## 📈 11. Advanced Relationship Visuals

- **Age vs BodyFat**: Colored by Sex with LOWESS regression.
- **BMI vs Abdomen**: Cluster-based visual to analyze abdominal obesity patterns.

---

## ✅ Summary of Insights Supported

- BMI and Waist-based ratios are strong body composition indicators.
- Sex and age play crucial roles in body fat distribution.
- Unsupervised clustering highlights natural groupings in biometric data.
- t-SNE, UMAP, and dendrograms confirm the internal structure of the dataset.

---
