from __future__ import annotations

import json
from typing import Sequence

import requests


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.ok
        except requests.RequestException:
            return False

    def generate(self, prompt: str, system: str | None = None) -> str | None:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if system:
            payload["system"] = system

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except (requests.RequestException, json.JSONDecodeError):
            return None

    def answer_from_context(
        self,
        query: str,
        contexts: Sequence[dict],
        history_text: str,
    ) -> str | None:
        numbered_context = "\n\n".join(
            f"[{idx}] Source: {item['doc_name']}\n{item['chunk_text']}"
            for idx, item in enumerate(contexts, start=1)
        )
        system = (
            "You are a concise retrieval assistant. Use only the provided context. "
            "If the answer is not in the context, say so clearly."
        )
        prompt = (
            f"Recent conversation history:\n{history_text or 'None'}\n\n"
            f"User question: {query}\n\n"
            f"Retrieved context:\n{numbered_context}\n\n"
            "Return a helpful answer in 4-8 sentences. End with a 'Sources:' line "
            "mentioning the document names you used."
        )
        return self.generate(prompt=prompt, system=system)

    def summarize(self, text: str) -> str | None:
        system = "You summarize content in 2-3 short bullet points."
        prompt = f"Summarize the following content:\n\n{text}"
        return self.generate(prompt=prompt, system=system)
