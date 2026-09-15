"""Add alert review workflow columns

Revision ID: 008_alert_review
Revises: 007_task_schedule
Create Date: 2026-09-13
"""
from alembic import op
import sqlalchemy as sa


revision = '008_alert_review'
down_revision = '007_task_schedule'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('alerts') as batch:
        batch.add_column(sa.Column(
            'review_status',
            sa.String(length=20),
            nullable=False,
            server_default='pending',
        ))
        batch.add_column(sa.Column('reviewed_by', sa.String(length=64), nullable=True))
        batch.add_column(sa.Column('reviewed_at', sa.DateTime(), nullable=True))
        batch.add_column(sa.Column('review_note', sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table('alerts') as batch:
        batch.drop_column('review_note')
        batch.drop_column('reviewed_at')
        batch.drop_column('reviewed_by')
        batch.drop_column('review_status')
