from pydantic import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI ML Template"
    API_V1_STR: str = "/api"

    # MODEL_PATH: str = "/app/app/model_files"
    MODEL_PATH: list = ["app/model_files/", "app/vocabs/vocab_lstm.pth"]

    class Config:
        case_sensitive = True


settings = Settings()
