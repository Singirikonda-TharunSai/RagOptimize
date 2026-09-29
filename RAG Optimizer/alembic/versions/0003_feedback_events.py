"""Add feedback_events table and performance indexes on eval tables.

Revision ID: 0003
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "feedback_events",
        sa.Column("id",             sa.Integer,     primary_key=True),
        sa.Column("query",          sa.Text,        nullable=False),
        sa.Column("strategy",       sa.String(50),  nullable=False),
        sa.Column("query_type",     sa.String(50),  nullable=False),
        sa.Column("signal_type",    sa.String(20),  nullable=False),
        sa.Column("signal_value",   sa.Float,       nullable=False),
        sa.Column("was_exploration",sa.Boolean,     nullable=False, server_default="false"),
        sa.Column("session_id",     sa.String(100), nullable=True),
        sa.Column("created_at",     sa.DateTime,    server_default=sa.func.now()),
    )
    # Aggregation query filters on these three columns — index prevents full table scan every 60 s
    op.create_index(
        "ix_feedback_strategy_type_created",
        "feedback_events", ["strategy", "query_type", "created_at"],
    )
    op.create_index(
        "ix_ragas_strategy_type_created",
        "ragas_scores", ["strategy", "query_type", "created_at"],
    )
    op.create_index(
        "ix_judge_strategy_type_created",
        "judge_scores", ["strategy", "query_type", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_judge_strategy_type_created",    table_name="judge_scores")
    op.drop_index("ix_ragas_strategy_type_created",    table_name="ragas_scores")
    op.drop_index("ix_feedback_strategy_type_created", table_name="feedback_events")
    op.drop_table("feedback_events")
