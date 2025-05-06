import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class RandomForestTrainer:
    def __init__(self, model_output_path: str = "models/random_forest_diabetes.pkl", 
                 n_estimators: int = 100, max_depth: int = 6, random_state: int = 42):
        self.model_output_path = model_output_path
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, X_train, y_train):
        print("Training RandomForest model...")
        print(f"Training RandomForest on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy:", acc)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        return acc

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        joblib.dump(self.model, self.model_output_path)
        print(f"Model saved to: {self.model_output_path}")

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        self.train(X_train, y_train)
        acc = self.evaluate(X_test, y_test)
        self.save_model()
        return self.model, acc
