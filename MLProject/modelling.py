import pandas as pd
import argparse
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ambil argumen dari MLproject
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="student_performance_processed.csv")
args = parser.parse_args()

# load data
df = pd.read_csv(args.data)
X = df.drop("G3", axis=1)
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)

# log ke MLflow (run udah aktif dari `mlflow run`)
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 10)
mlflow.log_metric("mse", mse)

# simpan model buat Docker
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    input_example=X_test.iloc[:5]
)

print(f"Training selesai - MSE: {mse}")
