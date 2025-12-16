from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

# final lab 3 model: breast cancer SVC with nested CV-tuned params
RUN_ID = "fa213ebb4f71423b9e357dc2335ffa66"

# use the run-based URI so mlflow knows exactly which model artifact to load
MODEL_URI = f"runs:/{RUN_ID}/model"

app = FastAPI(title="CancerClassificationAPI")

# load the trained model once at startup
model = mlflow.pyfunc.load_model(MODEL_URI)


class Features(BaseModel):
    # list of rows, each row is a list of feature values
    data: list[list[float]]


@app.get("/health")
def health():
    # simple health check
    return {"status": "healthy"}


@app.post("/predict")
def predict(features: Features):
    # turn incoming list-of-lists into a dataframe
    df = pd.DataFrame(features.data)

    # run the SVC pipeline from lab 3
    preds = model.predict(df)

    # return predictions as plain python types
    return {"predictions": preds.tolist()}
