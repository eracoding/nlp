from pydantic import BaseModel, Field, StrictStr
from typing import Optional


class PredictRequest(BaseModel):
    model: Optional[str] = Field(default='glove')
    search: Optional[str] = Field(default='Harry Potter')
    # brand: Optional[str] = Field(default='Unknown')
    # year: Optional[str] = Field(default='2000')
    # km_driven: Optional[str] = Field(default='0')
    # fuel: Optional[str] = Field(default='Petrol')
    # seller_type: Optional[str] = Field(default='Dealer')
    # transmission: Optional[str] = Field(default='manual')
    # owner: Optional[str] = Field(default='First')
    # mileage: Optional[str] = Field(default='0')
    # engine: Optional[str] = Field(default='Standard')
    # max_power: Optional[str] = Field(default='0')
    # seats: Optional[str] = Field(default='4')


class PredictResponse(BaseModel):
    # result: float = Field(..., title="result", description="Predict Value", example=0.9)
    result: StrictStr = Field(..., title="result", description="Model Inference", example="polutes")
