import os
import joblib
import numpy as np
import pandas as pd

from typing import Optional
from pydantic import BaseModel, Field, validator
import bentoml
from bentoml.io import JSON

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.joblib")
sk_model = joblib.load(MODEL_PATH)

class BuildingInput(BaseModel):
    property_gfa_total: float = Field(..., gt=0, description="Surface totale du bâtiment (pieds carrés)")
    year_built: int = Field(..., ge=1800, le=2100, description="Année de construction")
    number_of_floors: int = Field(..., ge=1, le=200, description="Nombre d'étages")

    primary_property_type: Optional[str] = None
    building_type: Optional[str] = None

    @validator("property_gfa_total")
    def check_gfa(cls, v):
        if v > 2_000_000:
            raise ValueError("property_gfa_total est trop grand pour un bâtiment normal.")
        return v

svc = bentoml.Service("seattle_energy_service")

@svc.api(input=JSON(pydantic_model=BuildingInput), output=JSON())
def predict(input_data: BuildingInput):
    feature_names = list(getattr(sk_model, "feature_names_in_", []))

    if feature_names:
        row = {name: np.nan for name in feature_names}

        mapping = {
            "PropertyGFATotal": input_data.property_gfa_total,
            "YearBuilt": input_data.year_built,
            "NumberofFloors": input_data.number_of_floors,
            "PrimaryPropertyType": input_data.primary_property_type,
            "BuildingType": input_data.building_type,
        }

        for col, val in mapping.items():
            if col in row and val is not None:
                row[col] = val

        df = pd.DataFrame([row])
    else:
        df = pd.DataFrame(
            [[input_data.property_gfa_total,
              input_data.year_built,
              input_data.number_of_floors]],
            columns=["property_gfa_total", "year_built", "number_of_floors"],
        )

    pred = sk_model.predict(df)[0]

    return {"predicted_site_energy_use": float(pred)}