from fastapi import FastAPI, HTTPException, Query
from fastapi import BackgroundTasks,  File, UploadFile
from typing import List, Optional
import yaml


import src.config_env
import src.main_tabular as da_tab
import src.main_time_series as da_ts
from src.server.schemas import (AugmentationRequest,  AugmentationRequestDummy, GenerateTimeSeriesRequest, 
                                AugmentedDataResponse, ModelsTimeSeriesInfoResponse, AugmentedDataTimeSeriesResponse)

app = FastAPI(
    title="Policy_Based_DA - I2cat",
    description="A service for policy-based data augmentation.",
    version="0.1.0",
    contact={
        "name": "Albert Calvo (i2cat)",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

tags_metadata = [
    {
        "name": "default",
        "description": "Operations with DA vanilla techniques.",
    }
]

 
@app.post("/augment_tabular_data", response_model=AugmentedDataResponse, tags=["default"])
async def augment_tabular_data(request: AugmentationRequest):
    """
    Augment data based on the dataset file path and method.
    
    Parameters:
 
       - dataset (str): Path to the dataset file.
       - method (str): The augmentation method to use (e.g., "SMOTE", "ADASYN", "CTGAN", "AUTO").
       - label_column_name (str): Name of the column containing the labels.
       - n_samples (int): Number of samples to generate.
       - distance (str): Type of distance to consider (e.g., "pairwise", "wasserstein").
    
    Returns:
       - AugmentedDataResponse: The augmented data and distances.
    """
    try:   
 
        data_path = request.data
        label = request.label_column_name
        n_samples = request.n_samples
        method = request.method
        distance = request.distance
 
        augmented_data = da_tab.augment_data(data_path, label, method, n_samples, distance, False)
         
        metadata = {"method": str(method),
                    "n_samples": str(n_samples), 
                    "distances": str(distance)}
        
        return AugmentedDataResponse(experiment_metadata=metadata, 
                                     augmented_data=augmented_data["augmented_data"].to_dict(orient='records'))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Bad Request: {str(ve)}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
 
 
@app.post("/augment_data_unittest", response_model=AugmentedDataResponse, tags=["default"])
async def augment_data_dummy(request: AugmentationRequestDummy) -> AugmentedDataResponse:
    """
    Augment data based on the dataset file path and method.

    Parameters:
       - method (str): The augmentation method to use (e.g., "SMOTE", "ADASYN", "CTGAN", "AUTO").
       - n_samples (int): Number of samples to generate.
       - distance (str): Type of distance to consider (e.g., "pairwise", "wasserstein").

    Returns:
       - AugmentedDataResponse: The augmented data and distances.
    """
    try:
 
        n_samples, method, distance = request.n_samples, request.method, request.distance    
        augmented_data = da_tab.augment_data(None, "target", method, n_samples, distance, False)
            
        #if distance == "pairwise":
        #    distance_result = augmented_data["distances"]["pairwise"]
        #else:
        #    distance_result = augmented_data["distances"]["wasserstein"]
            
        metadata = {"method": str(method),
                    "n_samples": str(n_samples), 
                    "distances": str(distance)}
        
        return AugmentedDataResponse(experiment_metadata=metadata, 
                                     augmented_data=augmented_data["augmented_data"].to_dict(orient='records'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/train_times_series_da_model")
async def train_times_series_da_model(config: UploadFile = File(...), 
                                      background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Augment data based on the dataset file path and method.
    - config: Configuration for the data augmentation model (YAML file)
 
    """
    try:
        contents = await config.read()
        
        try:
            config_data = yaml.safe_load(contents)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML file: {str(e)}")

        background_tasks.add_task(da_ts.train_model, config_data)

        # Return response
        return {"message": "Training started. It may take some time."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

 
 

@app.get("/get_DA_time_series_experiments", response_model=List[ModelsTimeSeriesInfoResponse])
async def get_DA_time_series_experiments(experiment_name: Optional[str] = Query(None, alias="experiment_name")):
    """
    Retrieve the different models and their status of a given experiment.
    - experiment_name (Optional): Name of the experiment to retrieve.
    """
    try:
        experiment_information = da_ts.get_experiment_runs(experiment_name)
        return experiment_information
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")



@app.post("/generate_time_series")
async def generate_time_series(request: GenerateTimeSeriesRequest):
    """
    Generate time-series samples based on a trained model.
    - run_id: The ID of the model run to use for inference.
    - n_samples: The number of samples to generate.
    """
    try:
        # Load the model and generate samples
        run_id = request.run_id
        n_samples = request.n_samples
        augmented_data = da_ts.generate_samples(run_id, n_samples)
        return AugmentedDataTimeSeriesResponse(augmented_data=augmented_data.tolist())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")