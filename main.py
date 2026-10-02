import os
import logging
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.graph import process_message
from app.tools.memory import reset_history
from app.tools.ingest import transcribe_audio, download_telegram_file, extract_text_from_document
from app.tools.rag import ingest_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

app = FastAPI(title="CTO Hub", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HELP_TEXT = """🤖 CTO Hub — Comandos y funciones

/reset — Borra el historial de conversación
/help — Esta ayuda

Cómo usarme:
- Hablame en lenguaje natural
- Mandame un audio → lo transcribo, proceso e indexo
- Mandame un documento o PDF → lo leo e indexo
- "registrá el 1:1 con Her: tema1, tema2"
- "anotá tarea: revisar PR de Zorro"
- "qué tareas tengo pendientes?"
- "toma nota del siguiente texto: [texto]"
- "qué sé sobre [tema]?" → busca en tu knowledge base
- "recordá que [hecho importante]"
"""

async def send_message(chat_id: str, text: str):
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{TELEGRAM_API}/sendMessage", json={
            "chat_id": chat_id,
            "text": text
        })
        logger.info(f"Telegram: {r.status_code}")

@app.get("/health")
def health():
    return {"status": "ok", "service": "cto-hub", "version": "3.0.0"}

@app.post("/telegram/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        message = data.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))

        if message.get("from", {}).get("is_bot", False):
            return {"ok": True}

        if not chat_id:
            return {"ok": True}

        text = message.get("text", "").strip()
        voice = message.get("voice") or message.get("audio")
        document = message.get("document")
        caption = message.get("caption", "").strip()

        # Comandos
        if text.lower() == "/reset":
            reset_history(chat_id)
            await send_message(chat_id, "🗑️ Historial borrado.")
            return {"ok": True}

        if text.lower() == "/help":
            await send_message(chat_id, HELP_TEXT)
            return {"ok": True}

        # Mensaje de voz
        if voice:
            async def handle_voice():
                await send_message(chat_id, "🎙️ Transcribiendo...")
                try:
                    audio_bytes = await download_telegram_file(voice["file_id"])
                    transcribed = await transcribe_audio(audio_bytes)
                    if not transcribed.strip():
                        await send_message(chat_id, "⚠️ No pude entender el audio.")
                        return
                    logger.info(f"Transcripción: {transcribed}")
                    await send_message(chat_id, f"📝 Entendí: {transcribed}")
                    # Ingestar en knowledge base
                    ingest_text(transcribed, source="audio", type="audio")
                    response = process_message(chat_id, transcribed)
                    await send_message(chat_id, response)
                except Exception as e:
                    logger.error(f"Error en voz: {e}", exc_info=True)
                    await send_message(chat_id, "⚠️ Error procesando el audio.")
            background_tasks.add_task(handle_voice)
            return {"ok": True}

        # Documento
        if document:
            async def handle_document():
                await send_message(chat_id, "📄 Leyendo el documento...")
                try:
                    file_bytes = await download_telegram_file(document["file_id"])
                    mime_type = document.get("mime_type", "text/plain")
                    filename = document.get("file_name", "documento.txt")
                    text_content = await extract_text_from_document(file_bytes, mime_type, filename)
                    # Ingestar en knowledge base
                    ingest_text(text_content, source=filename, type="document")
                    prompt = caption if caption else "Procesá este documento y decime de qué trata"
                    full_message = f"{prompt}\n\n[Contenido de '{filename}']:\n{text_content[:6000]}"
                    response = process_message(chat_id, full_message)
                    await send_message(chat_id, response)
                except Exception as e:
                    logger.error(f"Error en documento: {e}", exc_info=True)
                    await send_message(chat_id, "⚠️ Error procesando el documento.")
            background_tasks.add_task(handle_document)
            return {"ok": True}

        # Texto normal
        if text:
            async def handle_text():
                # Ingestar textos largos automáticamente
                if len(text) > 200:
                    ingest_text(text, source="telegram", type="text")
                response = process_message(chat_id, text)
                await send_message(chat_id, response)
            background_tasks.add_task(handle_text)
            return {"ok": True}

        return {"ok": True}

    except Exception as e:
        logger.error(f"Error en webhook: {e}", exc_info=True)
        return {"ok": True}
