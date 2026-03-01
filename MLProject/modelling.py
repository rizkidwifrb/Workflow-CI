import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===============================
# SET MLFLOW TRACKING URI
# ===============================
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# aktifkan autolog
mlflow.sklearn.autolog()

# ===============================
# PATH DATA (FIX SESUAI FOLDER LU)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(
    BASE_DIR,
    "dataset_preprocessing",
    "data_clean.csv"
)

print("Membaca data dari:", data_path)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(data_path)

print("Dataset berhasil dibaca")
print(df.head())

# ===============================
# FEATURE & TARGET
# ===============================
target_column = df.columns[-1]

X = df.drop(target_column, axis=1)
y = df[target_column]

# ubah jadi klasifikasi
y = (y > y.median()).astype(int)

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# TRAIN MODEL
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===============================
# EVALUASI
# ===============================
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))