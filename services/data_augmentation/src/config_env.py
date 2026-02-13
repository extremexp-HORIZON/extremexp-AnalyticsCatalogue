import os

# Set environment variables
os.environ['TMP_DIR'] = 'resources/tmp'
os.environ['DATA_DIR'] = 'resources/data'
os.environ['MLFLOW_URI'] = 'http://localhost:5005'
os.environ['API_URI'] = 'http://localhost:8000'