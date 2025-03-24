import pandas as pd
import joblib

# Load model
model = joblib.load("bodyfat_model.pkl")

def predict_bodyfat(data):
    '''Takes a DataFrame with input data and returns predictions.'''
    return model.predict(data)

# Example usage
if __name__ == "__main__":
    sample_data = pd.DataFrame([[85, 95, 100, 180, 60, 38, 32, 37]], 
                               columns=["Abdomen", "Chest", "Hip", "Weight", "Thigh", "Knee", "Biceps", "Neck"])
    prediction = predict_bodyfat(sample_data)
    print("Predicted body fat percentage:", prediction)
