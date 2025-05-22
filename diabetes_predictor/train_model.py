# diabetes_predictor/train_model.py

import pdb
import threading

import hydra
import mlflow
import mlflow.models
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from diabetes_predictor.data.make_dataset import load_arff_data, preprocess_data
from diabetes_predictor.models.model import RandomForestTrainer
from diabetes_predictor.utils.logging_config import get_logger, setup_logging
from monitoring.metrics_logger import run_metrics_server, update_metrics

# Set up logging
setup_logging()
logger = get_logger(__name__)
threading.Thread(target=run_metrics_server, daemon=True).start()  
#set up hydra
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    logger.info("Starting model training process")

    # Set MLflow experiment name
    mlflow.set_experiment("diabetes-prediction")

    # Start MLflow run
    with mlflow.start_run():
        logger.info("MLflow run started")

        # Log configuration parameters
        mlflow.log_param("n_estimators", cfg.model.n_estimators)
        mlflow.log_param("max_depth", cfg.model.max_depth)
        mlflow.log_param("random_state", cfg.seed)
        mlflow.log_param("test_size", cfg.data.test_size)
    
        logger.info("Loading and preprocessing data")
        df = load_arff_data(cfg.data.input_path)
        df = preprocess_data(df)
        logger.info(f"Data loaded and preprocessed. Shape: {df.shape}")
        
        X = df.drop(cfg.data.target_column, axis=1)
        y = df[cfg.data.target_column].astype(int)
        
        if cfg.debug:
            pdb.run("print(df.head())", globals(), locals())
            pdb.run("print(df.shape)", globals(), locals())
            pdb.run("print(cfg)", globals(), locals())
            pdb.run("print(X.columns.tolist())", globals(), locals())
            pdb.run("print(y.value_counts())", globals(), locals())

        logger.info("Splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            stratify=y,
            test_size=cfg.data.test_size,
            random_state=cfg.seed
        )
        logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

        logger.info("Initializing RandomForest trainer")
        trainer = RandomForestTrainer(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            random_state=cfg.seed
        )
    
        logger.info("Training and evaluating model")
        model, accuracy = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
        logger.info(f"Model training completed. Final accuracy: {accuracy:.4f}")

        # Make predictions to compute additional metrics
        y_pred = model.predict(X_test)

        # Prepare input example
        input_example = X_test.iloc[:5]

        # Infer model signature
        signature = infer_signature(input_example, model.predict(input_example))

        # Log the model with input example and signature
        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=input_example,
            signature=signature
        )

        # Log metrics to mlflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))

        
        logger.info("MLflow run completed")
        logger.info("Initializing RandomForest trainer")

        # Update Prometheus metrics
        update_metrics(acc=accuracy, loss=0.0)


if __name__ == "__main__":
    main()