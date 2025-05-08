# diabetes_predictor/train_model.py

from sklearn.model_selection import train_test_split

from diabetes_predictor.data.make_dataset import load_arff_data, preprocess_data
from diabetes_predictor.models.model import RandomForestTrainer


def main() -> None:
    df = load_arff_data("data/raw/diabetes.arff")
    df = preprocess_data(df)
    
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    trainer = RandomForestTrainer()
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()