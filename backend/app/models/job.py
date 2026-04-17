import uuid
from datetime import datetime
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
    sparse_ply: Optional[str] = None
    cameras_json: Optional[str] = None
    dense_ply: Optional[str] = None
    mesh_obj: Optional[str] = None
    mesh_stl: Optional[str] = None


PIPELINE_STAGES = ["validate", "features", "matching", "sfm", "dense", "export"]


class Job(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    image_count: int = 0
    images: List[str] = Field(default_factory=list)
    stages: Dict[str, StageResult] = Field(
        default_factory=lambda: {s: StageResult() for s in PIPELINE_STAGES}
    )
    error: Optional[str] = None
    artifacts: JobArtifacts = Field(default_factory=JobArtifacts)
    real_reconstruction: bool = False
