from sklearn.datasets import load_breast_cancer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict

import pandas as pd
import io

import main as da

app = FastAPI(
    title="Policy_Based_DA - I2cat",
    description="A service for policy-based data augmentation.",
    version="0.1.0",
    contact={
        "name": "Albert Calvo (i2cat)",
        "email": "albert.calvo@i2cat.net",
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

    
class AugmentationRequest(BaseModel):
    data_handler: str = Field(..., example="json")
    data: List[Dict[str, float]] = Field(..., example=[{"feature1": 0.5, "feature2": 1.2,"target":0},
                                                        {"feature1": 1.0, "feature2": 2.0, "target": 1},
                                                        {"feature1": 0.5, "feature2": 1.2,"target":1},
                                                        {"feature1": 1.0, "feature2": 5.0, "target": 1},
                                                        {"feature1": 6, "feature2": 1.2,"target":0},
                                                        {"feature1": 1.0, "feature2": 2.0, "target": 1}])
    method: str = Field(..., example="SMOTE")
    label_column_name: str = Field(..., example="label")
    n_samples: int = Field(..., example=20)
    distance: str = Field(..., example="pairwise")


class AugmentationRequestDummy(BaseModel): 
    method: str = Field(..., example="SMOTE")
    n_samples: int = Field(..., example=20)
    distance: str = Field(..., example="pairwise")


class AugmentedDataResponse(BaseModel):
    experiment_metadata: Dict[str, str]
    augmented_data: List[Dict[str, float]] 
    

#@app.post("/augment_data", tags=["default"])
@app.post("/augment_data", response_model=AugmentedDataResponse, tags=["default"])
async def augment_data(request: AugmentationRequest):
    """
    Augment data based on the dataset file path and method.
    
    Parameters:
       - request (AugmentationRequestPath): Contains the path to the dataset file and augmentation parameters.
       - dataset (str): Path to the dataset file.
       - method (str): The augmentation method to use (e.g., "SMOTE", "ADASYN", "CTGAN", "AUTO").
       - label_column_name (str): Name of the column containing the labels.
       - n_samples (int): Number of samples to generate.
       - distance (str): Type of distance to consider (e.g., "pairwise", "wasserstein").
    
    Returns:
       - AugmentedDataResponse: The augmented data and distances.
    """
    try:   
        data_handler = request.data_handler 
             
        if data_handler == "json": 
            data = pd.DataFrame(request.data)
        
        label = request.label_column_name
        n_samples = request.n_samples
        method = request.method
        distance = request.distance
        
        augmented_data = da.augment_data(data, label, method, n_samples, distance, False)
         
        metadata = {"method": str(method),
                    "n_samples": str(n_samples), 
                    "distances": str(distance)}
        
        return AugmentedDataResponse(experiment_metadata=metadata, 
                                     augmented_data=augmented_data["augmented_data"].to_dict(orient='records'))
    
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
        data = load_breast_cancer()
        dataset = pd.DataFrame(data.data, columns=data.feature_names)
        dataset["target"] = data.target
            
        n_samples, method, distance = request.n_samples, request.method, request.distance    
        augmented_data = da.augment_data(dataset, "target", method, n_samples, distance, False)
            
        if distance == "pairwise":
            distance_result = augmented_data["distances"]["pairwise"]
        else:
            distance_result = augmented_data["distances"]["wasserstein"]
            
        metadata = {"method": str(method),
                    "n_samples": str(n_samples), 
                    "distances": str(distance), 
                    "distance_result": str(distance_result)}

        return AugmentedDataResponse(experiment_metadata=metadata,
                                     augmented_data=augmented_data["augmented_data"].to_dict(orient='records'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
