from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Default to ~/.helium so the CLI works out of the box without any config.
    # Override with HELIUM_DATA_DIR=./data for the local dev server.
    data_dir: str = str(Path.home() / ".helium")

    min_audio_files: int = 1
    max_audio_files: int = 6
    max_audio_size_mb: int = 100
    target_speakers: int = 2

    diarization_backend: str = "pyannote/speaker-diarization"
    separation_backend: str = "speechbrain/sepformer-whamr"
    conversion_backend: str = "RedRepter/seed-vc-api"

    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "HELIUM_", "env_file": ".env"}


settings = Settings()
