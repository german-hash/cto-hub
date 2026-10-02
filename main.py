import os
import logging
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.graph import process_message
from app.tools.memory import reset_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

app = FastAPI(title="CTO Hub", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HELP_TEXT = """🤖 *CTO Hub — Comandos*

/reset — Borra el historial de conversación
/help — Esta ayuda

💬 *Cómo usarme:*
Hablame en lenguaje natural. Entiendo:
- Preguntas sobre tu equipo y proyectos
- "recordá que..." → guardo en memoria persistente
- Cualquier cosa relacionada a tu rol de CTO"""

async def send_message(chat_id: str, text: str):
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{TELEGRAM_API}/sendMessage", json={
            "chat_id": chat_id,
            "text": text
        })
        logger.info(f"Telegram: {r.status_code}")

@app.get("/health")
def health():
    return {"status": "ok", "service": "cto-hub"}

@app.post("/telegram/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        message = data.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))
        text = message.get("text", "").strip()

        # Ignorar mensajes del bot
        if message.get("from", {}).get("is_bot", False):
            return {"ok": True}

        if not chat_id or not text:
            return {"ok": True}

        if text.lower() == "/reset":
            reset_history(chat_id)
            await send_message(chat_id, "🗑️ Historial borrado.")
            return {"ok": True}

        if text.lower() == "/help":
            await send_message(chat_id, HELP_TEXT)
            return {"ok": True}

        async def handle():
            response = process_message(chat_id, text)
            await send_message(chat_id, response)

        background_tasks.add_task(handle)
        return {"ok": True}

    except Exception as e:
        logger.error(f"Error en webhook: {e}", exc_info=True)
        return {"ok": True}
