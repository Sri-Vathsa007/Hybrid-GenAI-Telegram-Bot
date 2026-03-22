from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    bot_mode: str = os.getenv("BOT_MODE", "hybrid").lower()
    db_path: Path = Path(os.getenv("DB_PATH", "data/bot.db"))
    docs_path: Path = Path(os.getenv("DOCS_PATH", "data/docs"))
    image_cache_dir: Path = Path(os.getenv("IMAGE_CACHE_DIR", "data/images"))

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    vision_model: str = os.getenv(
        "VISION_MODEL", "Salesforce/blip-image-captioning-base"
    )

    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2:3b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    top_k: int = int(os.getenv("TOP_K", "3"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    max_history: int = int(os.getenv("MAX_HISTORY", "3"))

    def prepare(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.docs_path.mkdir(parents=True, exist_ok=True)
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_rag_enabled(self) -> bool:
        return self.bot_mode in {"hybrid", "rag"}

    @property
    def is_vision_enabled(self) -> bool:
        return self.bot_mode in {"hybrid", "vision"}
