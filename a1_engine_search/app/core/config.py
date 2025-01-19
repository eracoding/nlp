from pydantic import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI ML Template"
    API_V1_STR: str = "/api"

    MODEL_PATH: str = "/app/app/model_files"

    class Config:
        case_sensitive = True


settings = Settings()
