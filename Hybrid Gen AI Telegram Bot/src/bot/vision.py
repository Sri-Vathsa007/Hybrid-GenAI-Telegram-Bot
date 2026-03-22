from __future__ import annotations

from pathlib import Path

from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from .database import Database
from .utils import extract_keywords


class VisionCaptioner:
    def __init__(self, model_name: str, db: Database):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.db = db

    def describe(self, user_id: int, image_path: Path) -> dict[str, str | list[str]]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        output_ids = self.model.generate(**inputs, max_new_tokens=40)
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
        tags = extract_keywords(caption, top_n=3)
        self.db.save_interaction(
            user_id=user_id,
            kind="image",
            user_input=str(image_path.name),
            bot_output=f"Caption: {caption}; Tags: {', '.join(tags)}",
        )
        return {"caption": caption, "tags": tags}
