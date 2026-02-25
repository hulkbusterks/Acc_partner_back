"""initial migration

Revision ID: 0001_initial
Revises: 
Create Date: 2025-11-20 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine import reflection
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def _has_column(conn, table_name, column_name):
    insp = reflection.Inspector.from_engine(conn)
    cols = [c['name'] for c in insp.get_columns(table_name)]
    return column_name in cols


def upgrade():
    conn = op.get_bind()
    # Add session.tone if missing (SQLite supports ALTER ADD COLUMN)
    if not _has_column(conn, 'session', 'tone'):
        op.add_column('session', sa.Column('tone', sa.Text(), nullable=True))

    # Add promptevent.prompt_question_id if missing
    if not _has_column(conn, 'promptevent', 'prompt_question_id'):
        op.add_column('promptevent', sa.Column('prompt_question_id', sa.Text(), nullable=True))


def downgrade():
    # downgrades for dropping columns are not reversible in SQLite easily; no-op
    pass
