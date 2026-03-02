import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# =====================
# SET EXPERIMENT
# =====================
mlflow.set_experiment("Student Performance CI")

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("../student_performance_preprocessing/student_performance_processed.csv")

print("Kolom dataset:")
print(df.columns)

# =====================
# SPLIT DATA
# =====================
X = df.drop("Performance Index", axis=1)
y = df["Performance Index"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# AUTOLOG
# =====================
mlflow.sklearn.autolog()

# =====================
# TRAIN MODEL
# =====================
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred) ** 0.5

# manual metric tambahan
mlflow.log_metric("rmse_manual", rmse)

print("Training CI selesai ✅")