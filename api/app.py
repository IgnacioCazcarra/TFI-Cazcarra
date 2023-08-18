import io
import pprint
from PIL import Image
from typing import Dict

from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body, FastAPI, status, UploadFile

from src.inference.inference_utils import api_prediction_wrapper, read_yaml, update_yaml, style_code


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="/TFI-Cazcarra/api/static"), name="static")

class HealthCheckResponse(BaseModel):
    status              : str = "OK"


class PredictRequest(BaseModel):
    img                 : UploadFile
    user_preferences    : Dict


class PredictResponse(BaseModel):
    code                : str

class PreferencesRequest(BaseModel):
    preferences         : str

templates = Jinja2Templates(directory="")

@app.get("/", summary="HTML home file")
async def home():
    return FileResponse("/TFI-Cazcarra/api/templates/home.html")


@app.get("/preferences", summary="HTML to let the user pick the model inference parameters.")
async def home():
    return FileResponse("/TFI-Cazcarra/api/templates/user_preferences.html")


@app.get("/healthcheck",
        summary="Perform a Health Check",
        response_description="Returns HTTP Status 200 (OK)",
        status_code=status.HTTP_200_OK,
        response_model=HealthCheckResponse
        )
async def health_check() -> HealthCheckResponse:
    return HealthCheckResponse(status="OK")


@app.post("/predict",
        summary="Create the SQL code from a ERD diagram image",
        response_description="Returns a string containing the SQL code",
        )
async def predict(img: UploadFile) -> PredictResponse:
    request_object_content = await img.read()
    img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
    sql_code = api_prediction_wrapper(img=img)
    sql_code = style_code(sql_code=sql_code)
    return PredictResponse(code=sql_code)


@app.get("/get_preferences",
        summary="Returns the YAML value for inference_params as a JSON",
        response_description="Returns a dict containing the values in inference_params.yaml",
        )
async def get_preferences():
    return read_yaml(yaml_path="/TFI-Cazcarra/inference_params.yaml")


@app.post("/update_preferences",
        summary="Updates the inference_params file with the preferred values of the user"
        )
async def update_preferences(preferences: Dict=Body()):
    update_yaml(data=preferences, yaml_path="/TFI-Cazcarra/inference_params.yaml")