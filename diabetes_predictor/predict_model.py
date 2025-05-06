from diabetes_predictor.models.predict import DiabetesPredictor
import pandas as pd

def main():
    input_data = pd.DataFrame([{
        "Pregnancies": 3,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 100,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.6,
        "Age": 32
    }])

    predictor = DiabetesPredictor()
    result = predictor.predict(input_data)

    print("Prediction Result:", result)

if __name__ == "__main__":
    main()
