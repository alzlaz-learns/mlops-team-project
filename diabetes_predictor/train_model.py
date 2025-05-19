# diabetes_predictor/train_model.py

from sklearn.model_selection import train_test_split

from diabetes_predictor.data.make_dataset import load_arff_data, preprocess_data
from diabetes_predictor.models.model import RandomForestTrainer
from diabetes_predictor.utils.logging_config import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)

def main() -> None:
    logger.info("Starting model training process")
    
    logger.info("Loading and preprocessing data")
    df = load_arff_data("data/raw/diabetes.arff")
    df = preprocess_data(df)
    logger.info(f"Data loaded and preprocessed. Shape: {df.shape}")
    
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"].astype(int)

    logger.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    logger.info("Initializing RandomForest trainer")
    trainer = RandomForestTrainer()
    
    logger.info("Training and evaluating model")
    model, accuracy = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    logger.info(f"Model training completed. Final accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()