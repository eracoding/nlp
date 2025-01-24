from typing import Any
from fastapi import Request, APIRouter
from fastapi.responses import HTMLResponse

from app.models.predict import PredictResponse, PredictRequest
from app.utils.process import make_three_words, deserialize

api_router = APIRouter()


@api_router.get("/", response_class=HTMLResponse)
async def get_form():
    with open("app/static/form.html", 'r') as file:
        return HTMLResponse(content=file.read())


@api_router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, payload: PredictRequest) -> Any:
    """
    ML Prediction API
    """
    input_data = payload.dict()

    model_name, input_text = make_three_words(input_data)
    model = request.app.state.model

    output = model.predict(model_name, input_text)

    result = deserialize(output, model_name, input_text)

    return PredictResponse(result=result)
