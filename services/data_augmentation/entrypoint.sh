
mlflow server \
  --backend-store-uri file:/app/resources/mlflow \
  --default-artifact-root file:/app/resources/mlflow \
  --host 0.0.0.0 \
  --port 5005 &


uvicorn src.server.server:app --host 0.0.0.0 --port 8000