FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install mlflow pandas scikit-learn numpy

CMD ["mlflow", "run", "."]