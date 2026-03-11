from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    redis_url: str

    newsapi_key: str = ""
    fred_api_key: str = ""
    finnhub_api_key: str = ""
    openrouter_api_key: str = ""

    mlflow_tracking_uri: str = ""
    mlflow_tracking_username: str = ""
    mlflow_tracking_password: str = ""

    prefect_api_key: str = ""
    prefect_api_url: str = ""

    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
