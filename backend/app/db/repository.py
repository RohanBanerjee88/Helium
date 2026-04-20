from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.job import Job
from .conversion import job_row_to_pydantic, pydantic_to_dict
from .models import JobRow


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, job: Job) -> Job:
        row = JobRow(**pydantic_to_dict(job))
        self._session.add(row)
        await self._session.flush()
        return job_row_to_pydantic(row)

    async def get(self, job_id: str) -> Optional[Job]:
        row = await self._session.get(JobRow, job_id)
        return job_row_to_pydantic(row) if row else None

    async def list_all(self) -> List[Job]:
        result = await self._session.execute(
            select(JobRow).order_by(JobRow.created_at.desc())
        )
        return [job_row_to_pydantic(r) for r in result.scalars()]

    async def save(self, job: Job) -> Job:
        data = pydantic_to_dict(job)
        data["updated_at"] = datetime.now(timezone.utc)
        row = await self._session.get(JobRow, job.id)
        if row is None:
            row = JobRow(**data)
            self._session.add(row)
        else:
            for k, v in data.items():
                setattr(row, k, v)
        await self._session.flush()
        return job_row_to_pydantic(row)

    async def delete(self, job_id: str) -> None:
        await self._session.execute(delete(JobRow).where(JobRow.id == job_id))

    async def list_ids(self) -> List[str]:
        result = await self._session.execute(select(JobRow.id))
        return list(result.scalars())
