import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf
from diabetes_predictor.train_model import main
from diabetes_predictor.data.make_dataset import load_arff_data, preprocess_data
from diabetes_predictor.models.model import RandomForestTrainer
from tests import _PATH_DATA

DATA_PATH = Path(_PATH_DATA) / "raw" / "diabetes.arff"

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing"""
    return OmegaConf.create({
        "model": {
            "n_estimators": 100,
            "max_depth": 6
        },
        "data": {
            "input_path": str(Path(_PATH_DATA) / "raw" / "diabetes.arff"),
            "test_size": 0.2,
            "target_column": "Outcome"
        },
        "seed": 42,
        "debug": False
    })

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data_path = Path(_PATH_DATA) / "raw" / "diabetes.arff"
    data = load_arff_data(data_path)
    return preprocess_data(data)

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Diabetes dataset not found")
def test_train_test_split(sample_data):
    """Test that train-test split maintains data distribution"""
    X = sample_data.drop('Outcome', axis=1)
    y = sample_data['Outcome']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )
    
    # Check split sizes
    expected_test_size = len(X) * 0.2
    assert np.isclose(len(X_test), expected_test_size, atol=1)  # Allow 1 sample difference
    
    # Check class distribution
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    assert np.allclose(train_dist, test_dist, atol=0.1)  # Allow 10% difference

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Diabetes dataset not found")
def test_forward_pass(sample_data):
    """Test that model can make predictions on new data"""
    X = sample_data.drop('Outcome', axis=1)
    y = sample_data['Outcome']
    
    trainer = RandomForestTrainer(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    
    # Train the model
    trainer.train(X, y)
    
    # Test forward pass on a single sample
    single_sample = X.iloc[0:1]
    prediction = trainer.model.predict(single_sample)
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]
    
    # Test forward pass on multiple samples
    multiple_samples = X.iloc[0:5]
    predictions = trainer.model.predict(multiple_samples)
    assert predictions.shape == (5,)
    assert all(pred in [0, 1] for pred in predictions)

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Diabetes dataset not found")
def test_model_training_and_evaluation(sample_data):
    """Test model training, prediction, and evaluation"""
    X = sample_data.drop('Outcome', axis=1)
    y = sample_data['Outcome']
    
    trainer = RandomForestTrainer(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    
    # Test model initialization and training
    assert hasattr(trainer, 'model')
    trainer.train(X, y)
    assert hasattr(trainer.model, 'predict')
    
    # Test predictions
    predictions = trainer.model.predict(X)
    assert len(predictions) == len(X)
    assert all(pred in [0, 1] for pred in predictions)
    
    # Test evaluation metrics
    accuracy = trainer.evaluate(X, y)
    assert 0 <= accuracy <= 1
    
    # Verify accuracy calculation
    manual_accuracy = (predictions == y).mean()
    assert np.isclose(accuracy, manual_accuracy)
    
    # Test that model can achieve high accuracy on training data
    assert accuracy > 0.8  # Model should achieve at least 80% accuracy on training data

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Diabetes dataset not found")
def test_invalid_input(sample_data):
    """Test that model handles invalid input appropriately"""
    X = sample_data.drop('Outcome', axis=1)
    y = sample_data['Outcome']
    trainer = RandomForestTrainer()
    
    # Test with mismatched lengths
    with pytest.raises(ValueError):
        trainer.train(X, y.iloc[:-1])  # y is one sample shorter than X

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Diabetes dataset not found")
@pytest.mark.parametrize("n_estimators,max_depth,min_accuracy", [
    (50, 3, 0.75),   # Small model
    (100, 6, 0.80),  # Medium model
    (200, 10, 0.85)  # Large model
])
def test_model_hyperparameters(sample_data, n_estimators, max_depth, min_accuracy):
    """Test model performance with different hyperparameters"""
    X = sample_data.drop('Outcome', axis=1)
    y = sample_data['Outcome']
    
    trainer = RandomForestTrainer(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # Train and evaluate
    trainer.train(X, y)
    accuracy = trainer.evaluate(X, y)
    
    # Verify accuracy meets minimum threshold for this configuration
    assert accuracy >= min_accuracy, f"Model with n_estimators={n_estimators}, max_depth={max_depth} achieved accuracy {accuracy:.3f}, expected at least {min_accuracy}"

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Diabetes dataset not found")
def test_train_model_main(tmp_path):
    """Test the main function in train_model.py using a temporary config and output directory."""
    # Create a temporary config file
    config = OmegaConf.create({
        "model": {
            "n_estimators": 100,
            "max_depth": 6
        },
        "data": {
            "input_path": str(DATA_PATH),
            "test_size": 0.2,
            "target_column": "Outcome"
        },
        "seed": 42,
        "debug": False
    })
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(config, config_path)

    # Load the configuration from the temporary file
    loaded_config = OmegaConf.load(config_path)

    # Call the main function with the loaded configuration
    main(loaded_config)

    # Verify that the model was saved (you can add more assertions as needed)
    model_path = Path("models/random_forest_diabetes.pkl")
    assert model_path.exists(), "Model file was not created" 