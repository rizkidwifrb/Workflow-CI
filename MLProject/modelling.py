import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="student_performance_processed.csv")
args = parser.parse_args()

df = pd.read_csv(args.data)

# TARGET SESUAI CSV LU
target_col = "Performance Index"
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)

# logging - run udah dibuat otomatis sama `mlflow run`
mlflow.log_param("model", "RandomForest")
mlflow.log_param("target", target_col)
mlflow.log_metric("mse", mse)

mlflow.sklearn.log_model(model, "model", input_example=X_test.iloc[:5])

print(f"Done - target={target_col}, MSE={mse:.4f}")
