# ===============================
# Import Library
# ===============================
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np


# ===============================
# Setup MLflow
# ===============================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Student Performance")


# ===============================
# Load Dataset (HASIL PREPROCESSING)
# ===============================
data_path = "student_performance_preprocessing/student_performance_processed.csv"

df = pd.read_csv(data_path)

print("Kolom dataset:")
print(df.columns)


# ===============================
# Split Feature & Target
# ===============================
X = df.drop("Performance Index", axis=1)
y = df["Performance Index"]


# ===============================
# Train Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ===============================
# Training Model + MLflow Logging
# ===============================
with mlflow.start_run():

    # Model
    model = LinearRegression()

    # Training
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # ===============================
    # Logging ke MLflow
    # ===============================
    mlflow.log_param("model_type", "LinearRegression")

    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Save model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

print("✅ Training selesai & model tersimpan di MLflow")