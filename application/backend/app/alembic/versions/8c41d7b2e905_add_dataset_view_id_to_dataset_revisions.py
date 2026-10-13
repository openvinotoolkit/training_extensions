# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""add_dataset_view_id_to_dataset_revisions

Revision ID: 8c41d7b2e905
Revises: 6fec06bf05e4
Create Date: 2026-09-16 09:12:44.115204

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8c41d7b2e905"
down_revision: str | Sequence[str] | None = "6fec06bf05e4"

branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the 'dataset_view_id' column to the dataset_revisions table."""
    # Deliberately not a foreign key: the value must outlive the dataset view it refers to.
    op.add_column("dataset_revisions", sa.Column("dataset_view_id", sa.Text(), nullable=True))


def downgrade() -> None:
    """Drop the 'dataset_view_id' column from the dataset_revisions table."""
    op.drop_column("dataset_revisions", "dataset_view_id")
