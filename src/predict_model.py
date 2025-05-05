# src/predict_model.py

import pandas as pd
import joblib
import os

# Load trained model
model_path = r"C:\Users\aleks\mlops-team-project\models\random_forest_diabetes.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = joblib.load(model_path)
print("? Model loaded successfully.")

# Example input: new patient data (values must match training feature order!)
new_data = pd.DataFrame([{
    "Pregnancies": 3,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 100,
    "BMI": 28.0,
    "DiabetesPedigreeFunction": 0.6,
    "Age": 32
}])

# Make prediction
prediction = model.predict(new_data)[0]
label = "Diabetic" if prediction == '1' else "Non-Diabetic"

print("?? Prediction Result:", label)
