import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from diabetes_predictor.models.model import RandomForestTrainer
from diabetes_predictor.data.make_dataset import load_arff_data, preprocess_data
from tests import _PATH_DATA

@pytest.fixture
def model():
    """Create a model instance for testing"""
    return RandomForestTrainer()

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data_path = Path(_PATH_DATA) / "raw" / "diabetes.arff"
    data = load_arff_data(data_path)
    processed_data = preprocess_data(data)
    X = processed_data.drop('Outcome', axis=1)
    y = processed_data['Outcome']
    return X, y

def test_model_output_shape(model, sample_data):
    """Test that model output has correct shape"""
    X, y = sample_data
    
    # Train the model
    model.train(X, y)
    
    # Get predictions
    predictions = model.model.predict(X)
    
    # Check output shape
    assert predictions.shape == (len(X),), "Predictions should be 1D array with same length as input"
    assert all(pred in [0, 1] for pred in predictions), "Predictions should be binary (0 or 1)"

def test_model_probability_shape(model, sample_data):
    """Test that model probability output has correct shape"""
    X, y = sample_data
    
    # Train the model
    model.train(X, y)
    
    # Get probability predictions
    proba = model.model.predict_proba(X)
    
    # Check output shape
    assert proba.shape == (len(X), 2), "Probability output should be (n_samples, 2)"
    assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities should sum to 1 for each sample"
