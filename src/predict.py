from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load MLflow model
model = mlflow.pyfunc.load_model("mlruns/384954395069818537/bece2c68ea6b45a0a97ff630770d5004/artifacts/titanic_model")

@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"survival_prediction": int(prediction[0])}
