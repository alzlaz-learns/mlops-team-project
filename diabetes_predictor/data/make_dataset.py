# diabetes_predictor\data\make_dataset.py
from pathlib import Path

import pandas as pd
from scipy.io import arff


# called by train_model.
def load_arff_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load and decode an ARFF dataset.

    Parameters:
    - filepath: Path to the ARFF file

    Returns:
    - DataFrame with properly decoded columns
    """
    data, _ = arff.loadarff(filepath)
    df = pd.DataFrame(data)

    # Decode byte strings (common in ARFF files)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: x.decode() if isinstance(x, bytes) else x)

    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize the dataset.

    Steps:
    - Replace placeholders or missing values with NaN
    - Drop rows with missing values
    - Convert all values to numeric
    - Normalize features (mean=0, std=1)
    """
    df = df.copy()

    # Replace missing NaN and drop them
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with any conversion issues
    df.dropna(inplace=True)

    # Normalize features
    if "Outcome" in df.columns:
        feature_cols = df.columns.drop("Outcome")
    else:
        feature_cols = df.columns

    df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()

    return df
