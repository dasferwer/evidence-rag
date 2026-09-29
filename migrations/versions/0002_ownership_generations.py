"""Ограничить доступ к базам и зафиксировать поколения индексации."""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "knowledge_bases",
        sa.Column("owner_id", sa.String(100), nullable=False, server_default="legacy"),
    )
    op.alter_column("knowledge_bases", "owner_id", server_default=None)
    op.drop_index("ix_knowledge_bases_slug", "knowledge_bases")
    op.drop_constraint("knowledge_bases_slug_key", "knowledge_bases", type_="unique")
    op.create_unique_constraint(
        "knowledge_bases_owner_slug_key", "knowledge_bases", ["owner_id", "slug"]
    )
    op.create_index("ix_knowledge_bases_owner", "knowledge_bases", ["owner_id", "name"])
    for column in [
        sa.Column("generation", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("indexed_generation", sa.Integer()),
        sa.Column("lease_token", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "next_attempt_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    ]:
        op.add_column("documents", column)
    op.execute("UPDATE documents SET indexed_generation=1 WHERE status='ready'")
    op.execute(
        "UPDATE outbox_events SET payload = payload || '{\"generation\": 1}'::jsonb "
        "WHERE event_type = 'document.ingest.requested' AND NOT payload ? 'generation'"
    )
    op.add_column(
        "chunks", sa.Column("generation", sa.Integer(), nullable=False, server_default="1")
    )
    op.create_index("ix_documents_due", "documents", ["status", "next_attempt_at"])
    op.create_index("ix_chunks_generation", "chunks", ["document_id", "generation"])
    op.create_index(
        "ix_chunks_lexical",
        "chunks",
        [sa.text("to_tsvector('simple', text)")],
        postgresql_using="gin",
    )


def downgrade() -> None:
    raise RuntimeError(
        "Откат удалит сведения о владельцах и поколениях; используйте исправляющую миграцию"
    )
