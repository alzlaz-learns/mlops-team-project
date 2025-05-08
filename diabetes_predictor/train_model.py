# diabetes_predictor/train_model.py

import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from diabetes_predictor.data.make_dataset import load_arff_data, preprocess_data
from diabetes_predictor.models.model import RandomForestTrainer


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Load and preprocess data
    df = load_arff_data(cfg.data.input_path)
    df = preprocess_data(df)

    X = df.drop(cfg.data.target_column, axis=1)
    y = df[cfg.data.target_column].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=cfg.data.test_size,
        random_state=cfg.seed
    )

    trainer = RandomForestTrainer(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        random_state=cfg.seed
    )

    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()