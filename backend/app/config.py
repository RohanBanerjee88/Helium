from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_dir: str = "./data"
    database_url: str = "cockroachdb+asyncpg://root@localhost:26257/helium"

    min_images: int = 8
    max_images: int = 20
    max_image_size_mb: int = 20

    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "HELIUM_", "env_file": ".env"}


settings = Settings()
