from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from .database import Database
from .llm import OllamaClient
from .utils import cosine_similarity, split_text


class MiniRAG:
    def __init__(
        self,
        docs_path: Path,
        db: Database,
        embedding_model_name: str,
        llm: OllamaClient,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        top_k: int = 3,
        max_history: int = 3,
    ):
        self.docs_path = docs_path
        self.db = db
        self.embedder = SentenceTransformer(embedding_model_name)
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.max_history = max_history

    def index_documents(self, rebuild: bool = False) -> int:
        if rebuild:
            self.db.clear_chunks()

        if self.db.chunk_count() > 0 and not rebuild:
            return self.db.chunk_count()

        indexed = 0
        for path in sorted(self.docs_path.glob("*")):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8").strip()
            chunks = split_text(
                text, chunk_size=self.chunk_size, overlap=self.chunk_overlap
            )
            if not chunks:
                continue
            embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                self.db.insert_chunk(path.name, idx, chunk, embedding)
                indexed += 1
        return indexed

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        scored: list[dict[str, Any]] = []
        for chunk in self.db.fetch_all_chunks():
            score = cosine_similarity(query_embedding, chunk["embedding"])
            scored.append({**chunk, "score": score})
        return sorted(scored, key=lambda x: x["score"], reverse=True)[: self.top_k]

    def _history_text(self, user_id: int) -> str:
        rows = self.db.recent_interactions(user_id, self.max_history)
        if not rows:
            return ""
        parts = []
        for row in rows:
            parts.append(f"User: {row['user_input']}")
            parts.append(f"Bot: {row['bot_output']}")
        return "\n".join(parts)

    @staticmethod
    def _fallback_answer(query: str, hits: list[dict[str, Any]]) -> str:
        if not hits:
            return (
                "I could not find relevant content in the local knowledge base. "
                "Try rephrasing the question or add more documents."
            )

        best = hits[0]
        supporting = " ".join(hit["chunk_text"][:220] for hit in hits[:2])
        return (
            f"Based on the knowledge base, here is the best answer to '{query}':\n\n"
            f"{supporting}\n\n"
            f"Primary source: {best['doc_name']}"
        )

    def answer(self, user_id: int, query: str) -> dict[str, Any]:
        cached = self.db.get_cached_query(query)
        if cached:
            response = {
                "answer": cached["response"],
                "sources": json.loads(cached["sources_json"]),
                "cached": True,
            }
            self.db.save_interaction(user_id, "ask", query, response["answer"])
            return response

        hits = self.retrieve(query)
        history_text = self._history_text(user_id)
        llm_answer = self.llm.answer_from_context(query, hits, history_text)
        answer = llm_answer or self._fallback_answer(query, hits)
        sources = [
            {
                "doc_name": hit["doc_name"],
                "chunk_index": hit["chunk_index"],
                "score": round(hit["score"], 4),
                "snippet": hit["chunk_text"][:180].strip(),
            }
            for hit in hits
        ]
        self.db.cache_query(query, answer, json.dumps(sources))
        self.db.save_interaction(user_id, "ask", query, answer)
        return {"answer": answer, "sources": sources, "cached": False}
