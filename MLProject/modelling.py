import argparse, pandas as pd, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
args = parser.parse_args()

df = pd.read_csv(args.data)
# target lu = Performance Index
X = df.drop("Performance Index", axis=1)
y = df["Performance Index"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("CI-Student-Performance")
with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")
    print(f"RMSE: {rmse}")
