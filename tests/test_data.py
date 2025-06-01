from pathlib import Path

import pandas as pd
import pytest

from diabetes_predictor.data.make_dataset import load_arff_data, preprocess_data
from tests import _PATH_DATA

EXPECTED_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
EXPECTED_FEATURE_COUNT = len(EXPECTED_FEATURES)

@pytest.fixture
def raw_data() -> pd.DataFrame:
    """Fixture to load raw data"""
    data_path = Path(_PATH_DATA) / "raw" / "diabetes.arff"
    if not data_path.exists():
        pytest.skip(f"Data file not found at {data_path}")
    try:
        return load_arff_data(data_path)
    except Exception as e:
        pytest.fail(f"Failed to load data: {str(e)}")

@pytest.fixture
def processed_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Fixture to load and preprocess data"""
    try:
        return preprocess_data(raw_data)
    except Exception as e:
        pytest.fail(f"Failed to preprocess data: {str(e)}")

def test_data_preprocessing(processed_data: pd.DataFrame) -> None:
    """Test that data is preprocessed correctly"""
    # Check basic properties
    assert isinstance(processed_data, pd.DataFrame), "Processed data should be a DataFrame"
    assert not processed_data.empty, "Processed data should not be empty"
    assert processed_data.shape[1] == EXPECTED_FEATURE_COUNT, f"Expected {EXPECTED_FEATURE_COUNT} features"
    
    # Check column names
    missing_cols = [col for col in EXPECTED_FEATURES if col not in processed_data.columns]
    assert not missing_cols, f"Missing columns: {missing_cols}"
    
    # Check data types
    for col in processed_data.columns:
        if col != 'Outcome':
            assert pd.api.types.is_numeric_dtype(processed_data[col]), f"Column {col} should be numeric"
        else:
            assert pd.api.types.is_numeric_dtype(processed_data[col]), "Outcome column should be numeric"

    # Check for missing values
    null_cols = processed_data.columns[processed_data.isnull().any()].tolist()
    assert not null_cols, f"Found null values in columns: {null_cols}"
    
    # Check normalization
    feature_cols = processed_data.columns.drop('Outcome')
    means = processed_data[feature_cols].mean()
    stds = processed_data[feature_cols].std()
    
    non_zero_mean = means[abs(means) > 1e-10]
    non_unit_std = stds[abs(stds - 1) > 1e-10]
    
    assert len(non_zero_mean) == 0, f"Features with non-zero mean: {non_zero_mean.index.tolist()}"
    assert len(non_unit_std) == 0, f"Features with non-unit std: {non_unit_std.index.tolist()}"

def test_train_test_split(processed_data: pd.DataFrame) -> None:
    """Test that data can be split into train and test sets"""
    from sklearn.model_selection import train_test_split
    
    X = processed_data.drop('Outcome', axis=1)
    y = processed_data['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test) 
