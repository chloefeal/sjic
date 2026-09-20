"""Add cameras.mount_position for pose view geometry

Revision ID: 009_camera_mount_position
Revises: 008_alert_review
Create Date: 2026-09-20
"""
from alembic import op
import sqlalchemy as sa


revision = '009_camera_mount_position'
down_revision = '008_alert_review'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('cameras') as batch:
        batch.add_column(sa.Column(
            'mount_position',
            sa.String(length=20),
            nullable=False,
            server_default='back_top',
        ))


def downgrade():
    with op.batch_alter_table('cameras') as batch:
        batch.drop_column('mount_position')
