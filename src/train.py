import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


def train(
    data_path: str = "data/breast_cancer.csv",
    C: float = 1.0,
    gamma: str = "scale",
) -> None:
    # check if we are inside an outer mlflow run (mlflow run .)
    parent_run = mlflow.active_run()

    # only set experiment if nothing is active (plain python call)
    if parent_run is None:
        mlflow.set_experiment("mlflow_lifecycle")

    # nested run if called from an mlflow project, top-level otherwise
    with mlflow.start_run(
        run_name="model_training_svc",
        nested=parent_run is not None,
    ):
        # load the dataset created by ingest.py
        df = pd.read_csv(data_path)
        X = df.drop(columns=["target"])
        y = df["target"]

        # stratified split to keep class balance similar
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # log key hyperparameters and data path
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("model_type", "svc_rbf")
        mlflow.log_param("C", C)
        mlflow.log_param("gamma", gamma)

        # this matches the lab 3 final pipeline:
        # standardscaler -> svc(rbf)
        pipe_svc = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=C,
                        gamma=gamma,
                        probability=True,  # needed for roc-auc
                        random_state=42,
                    ),
                ),
            ]
        )

        # fit on the training split only
        pipe_svc.fit(X_train, y_train)

        # use predicted probabilities for roc-auc
        preds_proba = pipe_svc.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds_proba)

        # log performance and the model artifact
        mlflow.log_metric("auc", auc)
        mlflow.sklearn.log_model(pipe_svc, "model")

        print(f"svc model trained, auc={auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/breast_cancer.csv",
        help="path to input csv with breast cancer data",
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="svc regularization parameter c",
    )

    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="svc gamma parameter (e.g. 'scale' or 'auto')",
    )

    args = parser.parse_args()

    train(
        data_path=args.data_path,
        C=args.C,
        gamma=args.gamma,
    )
