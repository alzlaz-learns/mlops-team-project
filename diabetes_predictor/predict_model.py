import pandas as pd

from diabetes_predictor.models.predict import DiabetesPredictor
from diabetes_predictor.utils.logging_config import get_logger, setup_logging

# Set up logging
setup_logging()
logger = get_logger(__name__)

def main() -> None:
    logger.info("Starting prediction process")
    
    logger.info("Creating sample input data")
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
    logger.info(f"Input data shape: {input_data.shape}")

    logger.info("Initializing DiabetesPredictor")
    predictor = DiabetesPredictor()
    
    logger.info("Making prediction")
    result = predictor.predict(input_data)
    logger.info(f"Prediction result: {result}")

if __name__ == "__main__":
    main()
