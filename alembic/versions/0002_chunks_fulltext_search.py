"""Add full-text search: a generated tsvector column on chunks with a GIN index.

Postgres GENERATED ALWAYS AS ... STORED keeps the tsvector column in sync
with ``content`` automatically on every insert/update — no ingest-time
computation, no trigger, no risk of the two drifting apart.

Text search configuration is 'english', not 'spanish': the seeded
historical-budgets corpus (see data/budgets_sample.json and the chunk
template in app/ingest/chunker.py) is entirely in English. The wrong
configuration would not raise an error — it would silently disable English
stemming and stopword filtering, degrading every full-text query without
any visible failure.
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """
    )
    op.create_index(
        "ix_chunks_content_tsv_gin", "chunks", ["content_tsv"], postgresql_using="gin"
    )


def downgrade():
    op.drop_index("ix_chunks_content_tsv_gin", table_name="chunks")
    op.drop_column("chunks", "content_tsv")
