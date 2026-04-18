"""create jobs table

Revision ID: 0001
Revises:
Create Date: 2026-04-18
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("image_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("images", JSONB, nullable=False, server_default="[]"),
        sa.Column("stages", JSONB, nullable=False, server_default="{}"),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("artifacts", JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "real_reconstruction", sa.Boolean, nullable=False, server_default="false"
        ),
    )
    op.create_index("idx_jobs_created_at", "jobs", [sa.text("created_at DESC")])
    op.create_index("idx_jobs_status", "jobs", ["status"])


def downgrade() -> None:
    op.drop_index("idx_jobs_status")
    op.drop_index("idx_jobs_created_at")
    op.drop_table("jobs")
