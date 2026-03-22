from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_name TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                UNIQUE(doc_name, chunk_index)
            );

            CREATE TABLE IF NOT EXISTS query_cache (
                query TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                sources_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                user_input TEXT NOT NULL,
                bot_output TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.commit()

    def clear_chunks(self) -> None:
        self.conn.execute("DELETE FROM chunks")
        self.conn.commit()

    @staticmethod
    def _pack_embedding(vector: np.ndarray) -> bytes:
        return np.asarray(vector, dtype=np.float32).tobytes()

    @staticmethod
    def _unpack_embedding(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def insert_chunk(
        self,
        doc_name: str,
        chunk_index: int,
        chunk_text: str,
        embedding: np.ndarray,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO chunks (doc_name, chunk_index, chunk_text, embedding)
            VALUES (?, ?, ?, ?)
            """,
            (doc_name, chunk_index, chunk_text, self._pack_embedding(embedding)),
        )
        self.conn.commit()

    def fetch_all_chunks(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT doc_name, chunk_index, chunk_text, embedding FROM chunks ORDER BY doc_name, chunk_index"
        ).fetchall()
        return [
            {
                "doc_name": row["doc_name"],
                "chunk_index": row["chunk_index"],
                "chunk_text": row["chunk_text"],
                "embedding": self._unpack_embedding(row["embedding"]),
            }
            for row in rows
        ]

    def chunk_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS total FROM chunks").fetchone()
        return int(row["total"])

    def get_cached_query(self, query: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT response, sources_json FROM query_cache WHERE query = ?",
            (query.strip().lower(),),
        ).fetchone()

    def cache_query(self, query: str, response: str, sources_json: str) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO query_cache (query, response, sources_json)
            VALUES (?, ?, ?)
            """,
            (query.strip().lower(), response, sources_json),
        )
        self.conn.commit()

    def save_interaction(
        self, user_id: int, kind: str, user_input: str, bot_output: str
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO interactions (user_id, kind, user_input, bot_output)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, kind, user_input, bot_output),
        )
        self.conn.commit()

    def recent_interactions(self, user_id: int, limit: int = 3) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT kind, user_input, bot_output, created_at
            FROM interactions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()[::-1]

    def last_interaction(self, user_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT kind, user_input, bot_output, created_at
            FROM interactions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
