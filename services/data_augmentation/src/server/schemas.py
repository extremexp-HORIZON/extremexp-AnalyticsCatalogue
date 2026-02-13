from typing import Dict, List, Optional
from pydantic import BaseModel, Field
 

class AugmentationRequest(BaseModel):

    data: str = Field(..., example="resources/data/tabular_data/data.pickle") 
    method: str = Field(..., example="SMOTE")
    label_column_name: str = Field(..., example="label")
    n_samples: int = Field(..., example=20)
    distance: str = Field(..., example="pairwise")


class AugmentationRequestDummy(BaseModel): 
    method: str = Field(..., example="SMOTE")
    n_samples: int = Field(..., example=20)
    distance: str = Field(..., example="pairwise")


class GenerateTimeSeriesRequest(BaseModel):
    run_id: str
    n_samples: int


class AugmentedDataResponse(BaseModel):
    experiment_metadata: Dict[str, str]
    augmented_data: List[Dict[str, float]] 


class AugmentedDataTimeSeriesResponse(BaseModel):
 
    augmented_data: List[List[List[float]]]


class ModelsTimeSeriesInfoResponse(BaseModel):
    experiment_name: str
    run_id: str
    run_name: Optional[str]
    status: str
    metrics: Dict[str, Optional[float]]


