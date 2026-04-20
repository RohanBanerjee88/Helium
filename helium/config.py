from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Default to ~/.helium so the CLI works out of the box without any config.
    # Override with HELIUM_DATA_DIR=./data for the local dev server.
    data_dir: str = str(Path.home() / ".helium")

    min_images: int = 8
    max_images: int = 20
    max_image_size_mb: int = 20

    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "HELIUM_", "env_file": ".env"}


settings = Settings()
