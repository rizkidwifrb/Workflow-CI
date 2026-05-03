FROM python:3.10-slim

WORKDIR /app
RUN pip install --no-cache-dir mlflow==2.12.1 scikit-learn pandas

COPY model /opt/ml/model

EXPOSE 8080
CMD ["mlflow", "models", "serve", "-m", "/opt/ml/model", "-h", "0.0.0.0", "-p", "8080"]
