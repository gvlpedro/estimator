"""Postgres DSN for the LangGraph checkpointer.

Reuses the same Postgres instance as the embeddings/pgvector tables
(``app.storage.db``) -- no new infrastructure. ``AsyncPostgresSaver`` is
built on ``psycopg`` (async v3), not ``asyncpg``, so the SQLAlchemy
``postgresql+asyncpg://`` URL is downgraded to a plain ``postgresql://`` DSN
that psycopg understands. The checkpointer creates its own tables
(``checkpoints``, ``checkpoint_writes``, ...) and coexists with the
embeddings schema without conflict.
"""

import os

from app.storage.db import DEFAULT_DATABASE_URL


def get_checkpointer_dsn() -> str:
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    return url.replace("postgresql+asyncpg://", "postgresql://")
