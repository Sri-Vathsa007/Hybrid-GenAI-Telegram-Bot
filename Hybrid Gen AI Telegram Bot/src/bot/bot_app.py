from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackContext,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import Settings
from .database import Database
from .llm import OllamaClient
from .rag import MiniRAG
from .utils import bullet_snippets
from .vision import VisionCaptioner

logger = logging.getLogger(__name__)


class HybridTelegramBot:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = Database(settings.db_path)
        self.llm = OllamaClient(settings.ollama_base_url, settings.llm_model)
        self.rag = (
            MiniRAG(
                docs_path=settings.docs_path,
                db=self.db,
                embedding_model_name=settings.embedding_model,
                llm=self.llm,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                top_k=settings.top_k,
                max_history=settings.max_history,
            )
            if settings.is_rag_enabled
            else None
        )
        self.vision = (
            VisionCaptioner(model_name=settings.vision_model, db=self.db)
            if settings.is_vision_enabled
            else None
        )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.help(update, context)

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = (
            "🤖 *Hybrid GenAI Telegram Bot*\n\n"
            "Commands:\n"
            "/ask <question> - Ask a question from the local knowledge base\n"
            "/image - Upload an image right after this command for captioning\n"
            "/summarize - Summarize your most recent bot interaction\n"
            "/help - Show this help message\n\n"
            "Examples:\n"
            "`/ask How do I reset MFA?`\n"
            "`/image` then send a photo"
        )
        await update.message.reply_markdown(message)

    async def ask(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self.rag:
            await update.message.reply_text("RAG mode is disabled for this bot.")
            return

        query = " ".join(context.args).strip()
        if not query:
            await update.message.reply_text("Usage: /ask <your question>")
            return

        await update.message.chat.send_action(action=ChatAction.TYPING)
        result = self.rag.answer(update.effective_user.id, query)
        sources_lines = [
            f"{item['doc_name']} (chunk {item['chunk_index']}, score={item['score']})"
            for item in result["sources"]
        ]
        cache_note = "\n\n⚡ Served from cache." if result["cached"] else ""
        reply = (
            f"{result['answer']}\n\n"
            f"*Source snippets used:*\n{bullet_snippets(sources_lines)}"
            f"{cache_note}"
        )
        await update.message.reply_markdown(reply)

    async def image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self.vision:
            await update.message.reply_text("Vision mode is disabled for this bot.")
            return
        context.user_data["awaiting_image"] = True
        await update.message.reply_text("Please upload an image now, and I will describe it.")

    async def summarize(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        last = self.db.last_interaction(update.effective_user.id)
        if not last:
            await update.message.reply_text("No interaction found yet. Use /ask or /image first.")
            return

        summary = self.llm.summarize(last["bot_output"])
        if not summary:
            summary = (
                f"• Last interaction type: {last['kind']}\n"
                f"• Your input: {last['user_input']}\n"
                f"• Bot output: {last['bot_output'][:220]}"
            )
        await update.message.reply_text(summary)

    async def handle_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self.vision:
            await update.message.reply_text("Vision mode is disabled for this bot.")
            return

        if not update.message.photo and not update.message.document:
            return

        if update.message.document and not (
            update.message.document.mime_type or ""
        ).startswith("image/"):
            return

        await update.message.chat.send_action(action=ChatAction.UPLOAD_PHOTO)
        photo = update.message.photo[-1] if update.message.photo else update.message.document
        file = await context.bot.get_file(photo.file_id)

        filename = (
            update.message.document.file_name
            if update.message.document and update.message.document.file_name
            else f"telegram_{update.effective_user.id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg"
        )
        target = Path(self.settings.image_cache_dir) / filename
        await file.download_to_drive(custom_path=str(target))

        result = self.vision.describe(update.effective_user.id, target)
        context.user_data["awaiting_image"] = False
        await update.message.reply_text(
            "🖼️ Caption: "
            f"{result['caption']}\n"
            f"🏷️ Tags: {', '.join(result['tags'])}"
        )

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if context.user_data.get("awaiting_image"):
            await update.message.reply_text("I'm waiting for an image upload, not text.")
            return

        await update.message.reply_text(
            "Use /ask for text questions or /image before sending a picture."
        )

    async def on_error(
        self, update: object, context: CallbackContext
    ) -> None:  # pragma: no cover
        logger.exception("Unhandled error", exc_info=context.error)

    def build_application(self) -> Application:
        if not self.settings.telegram_bot_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is missing. Copy .env.example to .env and set it."
            )

        if self.rag:
            self.rag.index_documents(rebuild=False)

        application = Application.builder().token(self.settings.telegram_bot_token).build()
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help))
        application.add_handler(CommandHandler("ask", self.ask))
        application.add_handler(CommandHandler("image", self.image))
        application.add_handler(CommandHandler("summarize", self.summarize))
        application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        application.add_handler(
            MessageHandler(filters.Document.IMAGE, self.handle_photo)
        )
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text)
        )
        application.add_error_handler(self.on_error)
        return application


def run() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )
    settings = Settings()
    settings.prepare()
    bot = HybridTelegramBot(settings)
    application = bot.build_application()
    application.run_polling(allowed_updates=Update.ALL_TYPES)
