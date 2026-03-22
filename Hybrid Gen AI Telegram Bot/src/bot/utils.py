from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "user",
    "image",
    "photo",
    "showing",
}


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def cosine_similarity(vec_a, vec_b) -> float:
    denom = (float((vec_a**2).sum()) ** 0.5) * (float((vec_b**2).sum()) ** 0.5)
    if denom == 0:
        return 0.0
    return float((vec_a @ vec_b) / denom)


def extract_keywords(text: str, top_n: int = 3) -> list[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", text.lower())
    words = [w for w in cleaned.split() if len(w) > 2 and w not in STOPWORDS]
    counts = Counter(words)
    return [word for word, _ in counts.most_common(top_n)]


def bullet_snippets(values: Iterable[str]) -> str:
    return "\n".join(f"• {value}" for value in values)
