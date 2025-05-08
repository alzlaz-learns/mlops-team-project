from pathlib import Path
from typing import Optional, Union

import joblib
import pandas as pd

from diabetes_predictor.data.make_dataset import preprocess_data


class DiabetesPredictor:
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        if model_path is None:
            # Relative to project root
            model_path = Path("models/random_forest_diabetes.pkl")
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model = joblib.load(model_path)
        
    def predict(self, input_df: pd.DataFrame) -> str:
        input_df = preprocess_data(input_df) # Preprocess the input data
        prediction = self.model.predict(input_df)[0]
        return "Diabetic" if int(prediction) == 1 else "Non-Diabetic"
