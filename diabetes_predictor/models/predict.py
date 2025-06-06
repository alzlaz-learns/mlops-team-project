# diabetes_predictor\models\predict.py
from pathlib import Path
from typing import Optional, Union

import joblib
import pandas as pd

from diabetes_predictor.data.make_dataset import preprocess_data
from diabetes_predictor.utils.logging_config import get_logger
from diabetes_predictor.utils.profiling import PerformanceTracker, profile_function

logger = get_logger(__name__)

class DiabetesPredictor:
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        if model_path is None:
            # Relative to project root
            model_path = Path("models/random_forest_diabetes.pkl")
        else:
            model_path = Path(model_path)

        logger.info(f"Initializing DiabetesPredictor with model path: {model_path}")
        self.performance_tracker = PerformanceTracker("DiabetesPredictor")

        if not model_path.exists():
            logger.error(f"Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        logger.info("Loading model from file")
        self.performance_tracker.start()
        self.model = joblib.load(model_path)
        duration = self.performance_tracker.end()
        self.performance_tracker.add_metric("model_load_time", duration)
        logger.info("Model loaded successfully")

    @profile_function
    def predict(self, input_df: pd.DataFrame) -> str:
        logger.info(f"Making prediction for input data with shape: {input_df.shape}")
        self.performance_tracker.start()

        input_df = preprocess_data(input_df)  # Preprocess the input data
        logger.info("Input data preprocessed")

        prediction = self.model.predict(input_df)[0]
        result = "Diabetic" if int(prediction) == 1 else "Non-Diabetic"

        duration = self.performance_tracker.end()
        self.performance_tracker.add_metric("prediction_time", duration)
        self.performance_tracker.add_metric("prediction", int(prediction))

        logger.info(f"Prediction result: {result}")
        return result
