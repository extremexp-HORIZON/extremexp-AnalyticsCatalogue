# Policy-based Data Augmentattion

## Introduction
This repository contains the code for a policy-based data augmentation task. The goal of this project 
is to provide a flexible and efficient way to augment datasets using policy-based approaches, enhancing 
the performance of machine learning models.

## Setting up API and MLFLOW
To facilitate the ezperimentation and development of the library we use the following tools.

1. Clone the Repository
```bash
git clone repo.git
cd policy-based-data-augmentation
```
2. run API from the folder policy-based-data-augmentation
```bash
uvicorn src.server.server:app --host 0.0.0.0 --port 8000 --reload
```

3. run mlflow server from the folder policy-based-data-augmentation
```bash
mlflow server \
  --backend-store-uri file:./resources/mlflow \
  --default-artifact-root file:./resources/mlflow \
  --host 0.0.0.0 \
  --port 5005
```
## Setting up Docker to run it as a service
To facilitate the deployment and usage of the data augmentation API, we provide a Docker setup. 
Follow these steps to set up the Docker environment:
1. Build the Docker image
```bash
docker build -t policybaseddataaugmentation .
```
2. Run the Docker container
```bash
docker run -d \
  --name data_augmentation_service \
  -p 9010:8000 \
  -p 5005:5005 \
  -v ./resources:/app/resources \
  --runtime=nvidia \
  -l is_api_service=true \
  policybaseddataaugmentation:latest
```
 

3. Debug the docker container
```bash
docker run -it -p 8000:8000 -p 5005:5005 \
  -v $(pwd)/policy-based-data-augmentation/resources:/app/resources \
  -e PYTHONPATH=/app/src:/app \
  policybaseddataaugmentation /bin/bash


  sudo docker exec -it policybaseddataaugmentation /bin/bash

 

This will expose the API on port 5005. Open your browser and navigate to http://localhost:5005 to 
check if the API is running successfully.


## API Documentation
The API provides a different endpoints to perform data augmentation. For detailed information 
on the available API endpoints and how to interact with the data augmentation service an 
API documentation is provided. To access the API documentation, go to http://localhost:5005/docs
after running the Docker container.

## Hands-On 
To demonstrate the usage of the code directly from the source, we have included a Python notebook 
(example.ipynb). Follow these steps to run the notebook:

1. Install the required packages
```bash
pip install -r requirements.txt
```
or alternatively, use the provided `poetry` environment:
```bash
poetry install
```

2. Start the Jupyter notebook
```bash
jupyter notebook
```
This will open a new tab in your browser. Navigate to the `/hands-on` examples and run the cells. 



