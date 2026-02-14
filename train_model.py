import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/Telco-Customer-Churn.csv")

df.drop("customerID", axis=1, inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df = pd.get_dummies(df)

X = df.drop("Churn", axis=1)
y = df["Churn"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

pickle.dump(model, open("model/model.pkl", "wb"))

pickle.dump(X.columns.tolist(), open("model/columns.pkl", "wb"))

print("model.pkl and columns.pkl created successfully")