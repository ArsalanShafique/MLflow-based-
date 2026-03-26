from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load latest model
model = mlflow.pyfunc.load_model("models:/house-price-model/1")

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict")
def predict(area: float, bedrooms: int):
    data = pd.DataFrame([[area, bedrooms]],
                        columns=["area", "bedrooms"])
    pred = model.predict(data)
    return {"prediction": float(pred[0])}
