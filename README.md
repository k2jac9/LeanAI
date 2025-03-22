# LeanAI
# 📌 Predicting Body Fat Percentage Using Machine Learning

## 1️⃣ Project Overview
This project utilizes the **Body Fat Prediction dataset** from Kaggle, which contains anthropometric measurements such as weight, height, waist circumference, and body fat percentage.

### 🔹 Business Objective
⚡ **Develop a machine learning model to predict body fat percentage** based on a person's physical measurements.

### 📌 Why is this important?
- **🏋️ Fitness & Health**: Personalized recommendations for diet and exercise.
- **🏥 Medical Diagnosis**: Assessment of obesity, diabetes, and cardiovascular disease risks.
- **🏆 Sports Industry**: Optimization of body composition for athletes.
- **💰 Insurance**: Health risk analysis for premium calculation.

From a business perspective, using anthropometric measurements like weight, height, and waist circumference to predict body fat percentage can have several practical and profitable motivations:

### 1. Health and Fitness Industry Applications
   - **Motivation**: The ability to estimate body fat percentage without expensive equipment (like DEXA scans or hydrostatic weighing) makes health assessments more accessible and cost-effective.
   - **Business Use Case**: Fitness companies (e.g., gyms, personal training services, or wearable tech firms like Fitbit or Apple) could integrate this into their offerings. For example, a gym could use a simple app or kiosk where clients input their measurements to get an instant body fat estimate, enhancing customer engagement and retention.
   - **Value Proposition**: Low-cost, scalable health insights encourage users to track progress, purchase subscriptions, or invest in personalized training plans.

### 2. Cost Reduction and Scalability
   - **Motivation**: Traditional body fat measurement methods are time-consuming, expensive, and require trained personnel or specialized tools. Anthropometric predictions bypass these barriers.
   - **Business Use Case**: A startup could develop a SaaS (Software as a Service) platform targeting small clinics, schools, or corporate wellness programs, providing affordable body fat analysis tools based on these measurements.
   - **Value Proposition**: Businesses save money while offering a valuable service, creating a competitive edge over rivals relying on pricier alternatives.

### 3. Personalized Product Offerings
   - **Motivation**: Body fat percentage is a key metric for tailoring nutrition plans, workout regimes, or even apparel sizing.
   - **Business Use Case**: E-commerce platforms (e.g., meal kit services like HelloFresh or fitness apparel brands like Lululemon) could use this data to recommend products suited to a customer’s body composition, increasing sales conversion rates.
   - **Value Proposition**: Enhanced personalization drives customer satisfaction and loyalty, boosting revenue through targeted upselling.

### 4. Public Health and Insurance Insights
   - **Motivation**: Body fat percentage is a better indicator of health risks (e.g., obesity-related diseases) than BMI alone, which can misclassify muscular individuals.
   - **Business Use Case**: Insurance companies or public health organizations could use this predictive model to assess risk profiles at scale, adjusting premiums or designing intervention programs.
   - **Value Proposition**: More accurate risk assessment reduces payouts for insurers or improves outcomes for health initiatives, saving costs long-term.

---

## 2️⃣ Data
📂 **Data Source**: [Kaggle: Body Fat Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset)  
📊 **Format**: CSV file  
💾 **Dataset Size**: 252 samples.

### 🔢 **Features in the dataset**
- `Density` – Body density (measured by hydrostatic weighing).
- `BodyFat` – **Body fat percentage (target variable)**.
- `Age` – Age.
- `Weight` – Weight (in pounds).
- `Height` – Height (in inches).
- `Neck`, `Chest`, `Abdomen`, `Hip`, `Thigh`, `Knee`, `Ankle`, `Biceps`, `Forearm`, `Wrist` – Circumference measurements of different body parts (in inches).

--- 

## 📌 FastAPI Backend  
This project includes a **FastAPI-based application** for predicting body fat percentage. The API provides both:  
- A **web form** for manual input  
- A **REST API** for external integration  

🔗 **[Full API Documentation](Project/api/README.md)**

 
### 🚀 Quick Start  
#### Run the API using Docker  
```bash
docker-compose up --build  # First time setup
docker-compose up          # Subsequent runs
```

- **Access the web interface**: [http://localhost:8000](http://localhost:8000)  
- **API documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)  

### 🔹 Example API request  
```bash
curl -X POST "http://localhost:8000/predict/" \
     -H "accept: application/json" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "abdomen=110&hip=120&weight=100&thigh=190&knee=50&biceps=38&neck=45"
```

For detailed instructions on deployment, troubleshooting, and advanced configurations, check out the **[API README](Project/api/README.md)**.


## 3️⃣ Methodology 
**Exploratory Data Analysis**

Our initial exploratory steps were the standard tasks of determining the number of features in the dataset, identifying any null or missing values, observing the scale of individual features and the data types we had to work with.  The dataset, although small, was very clean: no missing or null values, only floating and integer data types, and a total of 14 features.  A couple of important notes: the data was taken of men only, density was measured using under water weighing, and the remaiming features were measured using the measurement standards listed in Benhke and Wilmore (1974), pp. 45-48.

Matplotlib was used to visualize the distribution of the dataset, providing a deeper understanding of the spread of each feature. 

![image](https://github.com/user-attachments/assets/8b989424-30e2-48fc-9cae-b89ac1b0c5c4)




Most features follow a normal distribution, except for height, hip, and ankle, which exhibit slight skewness. 
A heatmap was applied to determine the correlation between different features, revealing a strong negative correlation between body fat and density. Additionally, weight shows a strong positive correlation with hip, chest, and abdomen size.  

![image](https://github.com/user-attachments/assets/4713c904-67c4-486b-ac84-00abfbf3a7a8)


We also utilized boxplots to identify outliers.

![image](https://github.com/user-attachments/assets/f7a5dfd3-0e4d-48c8-a8b6-877d01d2b68a)




Since our dataset initially contained only male measurements, which is a limition,  we extended our analysis to include female samples for a more comprehensive evaluation.
Exploratory Data Analysis (EDA), particularly through heat maps, revealed significant sex-based differences in body measurements. In our dataset, we encoded females as "1" and males as "0." We observed strong negative correlations for certain measurements, including Neck (-0.84), Forearm (-0.81), Wrist (-0.81), Chest (-0.71), Abdomen (-0.78), and Weight (-0.67). These findings indicate that males generally have larger body measurements than females. However, when assessing the influence of sex on body fat prediction, the effect appeared minimal (0.17).
Abdomen circumference emerged as a key indicator of body fat percentage, showing strong correlations with Body Fat (0.36), Chest (0.92), Hip (0.68), and Thigh (0.85).
Weight, on the other hand, was more strongly associated with skeletal and muscular body measurements rather than body fat alone. It exhibited high correlations with Chest (0.91), Abdomen (0.93), Hip (0.81), and Thigh (0.91), suggesting that weight increases in proportion to overall body dimensions.

Overall, the dataset highlights strong relationships between various body measurements, making it valuable for predictive modeling in health and fitness.

![image](https://github.com/user-attachments/assets/c816a08d-d563-433d-beb5-e7f5fe8ce83e)





**Data Processing**
Since the time limtation and the dataset is relatively simple and small, we decide to use as it is to start with a simple linear regression model initiatlly 


**Model Development**
- Due to small dataset size, started with simple linear regression model

![image](https://github.com/user-attachments/assets/6c1119d4-b359-4380-be0b-7fe80eabe1a3)

 
### 🛠 **Model evaluation metrics**
- **📉 MAE (Mean Absolute Error)** – Average absolute error.
- **📉 RMSE (Root Mean Squared Error)** – Root mean square error.
- **📈 R² (R-squared)** – Measures how well the model explains the variance in the data.

**Feature Selection**

## 4️⃣ Machine Learning Solution 
💡 **Potential algorithms to use**:
- ✔️ **Linear Regression** (for interpretability)
- ✔️ **Random Forest** (for high accuracy)
- ✔️ **XGBoost** (for advanced optimization)
- ✔️ **Neural Networks** (for complex relationships)

---

## 5️⃣ Expected Outcomes
✅ **Develop** a machine learning model that accurately predicts body fat percentage.  
✅ **Visualize** correlations between body measurements and fat percentage.  
✅ **Optimize** models and improve prediction accuracy.  

---

🚀 **This project will help individuals monitor their health, make predictions, and make informed decisions!** 🎯

Team Members:
Igor Bak
Alejandro Castellanos
Faisal Khan
Hassan Saade
Anna W
