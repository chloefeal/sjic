"""Add task daily schedule columns

Revision ID: 007_task_schedule
Revises: 006_drop_camera_role
Create Date: 2026-04-13
"""
from alembic import op
import sqlalchemy as sa


revision = '007_task_schedule'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('tasks') as batch:
        batch.add_column(sa.Column('schedule_start', sa.String(length=8), nullable=True))
        batch.add_column(sa.Column('schedule_end', sa.String(length=8), nullable=True))
        batch.add_column(sa.Column('schedule_paused', sa.Boolean(), nullable=True, server_default=sa.false()))


def downgrade():
    with op.batch_alter_table('tasks') as batch:
        batch.drop_column('schedule_paused')
        batch.drop_column('schedule_end')
        batch.drop_column('schedule_start')
