from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "student.csv"
MODEL_PATH = ROOT_DIR / "app" / "model.pkl"
MLFLOW_EXPERIMENT_NAME = "student-performance"
RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(DATA_PATH)
    features = dataset[["study_hours", "attendance", "previous_marks"]].astype(float)
    target = dataset["pass"]
    return features, target


def train_model() -> dict[str, float]:
    features, target = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    signature = infer_signature(x_train, model.predict(x_train))
    input_example = x_train.iloc[:1]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(classification_report(y_test, predictions), "classification_report.txt")
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            pip_requirements=[
                "mlflow>=2.15,<3.0",
                "scikit-learn>=1.5,<2.0",
                "pandas>=2.2,<3.0",
                "joblib>=1.4,<2.0",
            ],
        )

    return {"accuracy": accuracy}


if __name__ == "__main__":
    metrics = train_model()
    print(f"Model saved to {MODEL_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
