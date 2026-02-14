from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

model = pickle.load(open("../model/model.pkl", "rb"))

training_columns = pickle.load(open("../model/columns.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Customer Churn API Running"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    df = pd.get_dummies(df)

    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[training_columns]

    prediction = model.predict(df)[0]

    if prediction == 1:
        return {"prediction": "Customer will churn"}
    else:
        return {"prediction": "Customer will stay"}