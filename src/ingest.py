import os

import mlflow
from sklearn.datasets import load_breast_cancer
import pandas as pd


def ingest(output_path: str = "data/breast_cancer.csv") -> None:
    # check if there is already an active run
    parent_run = mlflow.active_run()

    # if there is no active run (direct python call), choose our experiment
    if parent_run is None:
        mlflow.set_experiment("mlflow_lifecycle")

    # if parent_run is not none, we are inside mlflow run → use nested run
    with mlflow.start_run(
        run_name="data_ingestion",
        nested=parent_run is not None,
    ):
        # load breast cancer dataset from sklearn
        data = load_breast_cancer(as_frame=True)
        df = data.frame

        # make sure the data folder exists
        os.makedirs("data", exist_ok=True)

        # save dataset as a csv file
        df.to_csv(output_path, index=False)

        # log a simple param (row count) and the csv as an artifact
        mlflow.log_param("rows", len(df))
        mlflow.log_artifact(output_path)

        print("data saved to:", output_path)


if __name__ == "__main__":
    ingest()
