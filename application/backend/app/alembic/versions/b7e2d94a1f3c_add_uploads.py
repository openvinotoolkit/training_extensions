# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""add_uploads

Revision ID: b7e2d94a1f3c
Revises: 8c41d7b2e905
Create Date: 2026-09-30 16:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7e2d94a1f3c"
down_revision: str | Sequence[str] | None = "8c41d7b2e905"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the 'uploads' table tracking resumable (TUS) uploads."""
    op.create_table(
        "uploads",
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=True),
        sa.Column("size", sa.BigInteger(), nullable=True),
        sa.Column("offset", sa.BigInteger(), nullable=False),
        sa.Column("state", sa.String(length=20), nullable=False),
        sa.Column("checksum", sa.String(length=255), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_uploads_state", "uploads", ["state"], unique=False)
    op.create_index("idx_uploads_expires_at", "uploads", ["expires_at"], unique=False)


def downgrade() -> None:
    """Drop the 'uploads' table."""
    op.drop_index("idx_uploads_expires_at", table_name="uploads")
    op.drop_index("idx_uploads_state", table_name="uploads")
    op.drop_table("uploads")
