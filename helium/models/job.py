import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageResult(BaseModel):
    status: StageStatus = StageStatus.PENDING
    message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    artifacts: List[str] = Field(default_factory=list)


class JobArtifacts(BaseModel):
    validation_report: Optional[str] = None
    diarization_output: Optional[str] = None
    separation_output: Optional[str] = None
    conversion_output: Optional[str] = None
    evaluation_report: Optional[str] = None
    export_manifest: Optional[str] = None


PIPELINE_STAGES = ["validate", "diarize", "separate", "convert", "evaluate", "export"]


class Job(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    audio_count: int = 0
    audio_files: List[str] = Field(default_factory=list)
    stages: Dict[str, StageResult] = Field(
        default_factory=lambda: {s: StageResult() for s in PIPELINE_STAGES}
    )
    error: Optional[str] = None
    artifacts: JobArtifacts = Field(default_factory=JobArtifacts)
    target_speakers: int = 2
    model_outputs_ready: bool = False
