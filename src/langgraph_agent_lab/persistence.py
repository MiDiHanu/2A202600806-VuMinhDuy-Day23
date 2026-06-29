"""Checkpointer adapter.

Provides LangGraph checkpointers for persistence/recovery. The default is an
in-memory saver (offline, CI-friendly). The SQLite backend persists state to
disk so a thread can be resumed after the process exits — the basis for the
crash-recovery / time-travel extensions.
"""

from __future__ import annotations

from typing import Any

DEFAULT_SQLITE_PATH = "checkpoints.db"


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    - ``none``     -> no checkpointer
    - ``memory``   -> MemorySaver (ephemeral, default)
    - ``sqlite``   -> SqliteSaver backed by a WAL-mode SQLite file on disk
    - ``postgres`` -> PostgresSaver (optional extension, needs DATABASE_URL)
    """
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if kind == "sqlite":
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        path = database_url or DEFAULT_SQLITE_PATH
        # check_same_thread=False: LangGraph may touch the connection from a
        # worker thread. WAL mode allows concurrent readers + durable writes.
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        saver = SqliteSaver(conn=conn)
        saver.setup()
        return saver

    if kind == "postgres":
        if not database_url:
            raise ValueError("postgres checkpointer requires database_url (DATABASE_URL)")
        from langgraph.checkpoint.postgres import PostgresSaver

        saver = PostgresSaver.from_conn_string(database_url)
        saver.setup()
        return saver

    raise ValueError(f"Unknown checkpointer kind: {kind}")
