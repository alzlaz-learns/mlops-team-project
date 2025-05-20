# diabetes_predictor\models\model.py
import os
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from diabetes_predictor.utils.logging_config import get_logger
from diabetes_predictor.utils.profiling import profile_function, PerformanceTracker

logger = get_logger(__name__)

class RandomForestTrainer:
    def __init__(self, model_output_path: str = "models/random_forest_diabetes.pkl", 
                 n_estimators: int = 100, max_depth: int = 6, random_state: int = 42):
        self.model_output_path = model_output_path
        logger.info(f"Initializing RandomForestClassifier with n_estimators={n_estimators}, max_depth={max_depth}")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.performance_tracker = PerformanceTracker("RandomForestTrainer")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        logger.info(f"Training RandomForest model on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        self.performance_tracker.start()
        self.model.fit(X_train, y_train)
        duration = self.performance_tracker.end()
        self.performance_tracker.add_metric("training_samples", X_train.shape[0])
        self.performance_tracker.add_metric("features", X_train.shape[1])
        logger.info("Model training completed")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        logger.info("Evaluating model on test set")
        self.performance_tracker.start()
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        duration = self.performance_tracker.end()
        self.performance_tracker.add_metric("accuracy", acc)
        self.performance_tracker.add_metric("test_samples", X_test.shape[0])
        logger.info(f"Model accuracy: {acc:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        return acc

    def save_model(self) -> None:
        logger.info(f"Saving model to {self.model_output_path}")
        self.performance_tracker.start()
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        joblib.dump(self.model, self.model_output_path)
        duration = self.performance_tracker.end()
        self.performance_tracker.add_metric("model_size_mb", os.path.getsize(self.model_output_path) / (1024 * 1024))
        logger.info("Model saved successfully")

    @profile_function
    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[RandomForestClassifier, float]:
        logger.info("Starting train and evaluate process")
        self.train(X_train, y_train)
        acc = self.evaluate(X_test, y_test)
        self.save_model()
        logger.info("Train and evaluate process completed")
        return self.model, acc
