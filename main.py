import os
import logging
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

load_dotenv()

from app.graph import process_message
from app.tools.memory import reset_history
from app.tools.ingest import transcribe_audio, download_telegram_file, extract_text_from_document
from app.tools.rag import ingest_text
from app.tools.calendar import get_auth_url, exchange_code, save_tokens
from app.tools.notion_import import run_notion_import
from app.tools.calendar_tools import set_current_chat_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Mapeo temporal code → chat_id para OAuth
_oauth_state: dict = {}

app = FastAPI(title="CTO Hub", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HELP_TEXT = """🤖 CTO Hub — Comandos

/reset — Borra historial
/conectar_calendar — Conecta Google Calendar
/importar_notion — Importa historial de Notion a la knowledge base
/help — Esta ayuda

Funciones:
- Registrar 1:1s, notas, tareas
- Buscar en knowledge base
- Ver agenda del calendario
- Procesar audios y documentos
- "recordá que [hecho]"
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
    return {"status": "ok", "service": "cto-hub", "version": "4.0.0"}

@app.get("/auth/google/callback")
async def google_callback(code: str, state: str = ""):
    """Callback de OAuth de Google."""
    chat_id = _oauth_state.get(state, "")
    if not chat_id:
        return HTMLResponse("<h2>Error: sesión expirada. Intentá de nuevo desde Telegram.</h2>")
    try:
        tokens = await exchange_code(code)
        save_tokens(chat_id, tokens)
        await send_message(chat_id, "✅ Google Calendar conectado. Probá con: 'qué tengo en el calendario esta semana?'")
        return HTMLResponse("<h2>✅ Google Calendar conectado. Podés cerrar esta ventana.</h2>")
    except Exception as e:
        logger.error(f"Error OAuth: {e}")
        return HTMLResponse(f"<h2>Error al conectar: {str(e)}</h2>")

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

        # Setear chat_id para las calendar tools
        set_current_chat_id(chat_id)

        if text.lower() == "/reset":
            reset_history(chat_id)
            await send_message(chat_id, "🗑️ Historial borrado.")
            return {"ok": True}

        if text.lower() == "/help":
            await send_message(chat_id, HELP_TEXT)
            return {"ok": True}

        if text.lower() == "/conectar_calendar":
            import uuid
            state = str(uuid.uuid4())
            _oauth_state[state] = chat_id
            auth_url = get_auth_url() + f"&state={state}"
            await send_message(chat_id, f"📅 Hacé click para conectar Google Calendar:\n{auth_url}")
            return {"ok": True}

        if text.lower() == "/importar_notion":
            async def handle_import():
                await send_message(chat_id, "📥 Importando Notion... puede tardar 1-2 minutos.")
                try:
                    results = run_notion_import()
                    ok = ", ".join(results["success"])
                    failed = ", ".join(results["failed"]) if results["failed"] else "ninguna"
                    await send_message(chat_id, f"✅ Importación completa\n• Páginas: {len(results['success'])}\n• Chunks: {results['chunks_total']}\n• Fallidas: {failed}")
                except Exception as e:
                    logger.error(f"Error importando Notion: {e}", exc_info=True)
                    await send_message(chat_id, f"⚠️ Error en la importación: {str(e)}")
            background_tasks.add_task(handle_import)
            return {"ok": True}

        # Voz
        if voice:
            async def handle_voice():
                await send_message(chat_id, "🎙️ Transcribiendo...")
                try:
                    audio_bytes = await download_telegram_file(voice["file_id"])
                    transcribed = await transcribe_audio(audio_bytes)
                    if not transcribed.strip():
                        await send_message(chat_id, "⚠️ No pude entender el audio.")
                        return
                    await send_message(chat_id, f"📝 Entendí: {transcribed}")
                    ingest_text(transcribed, source="audio", type="audio")
                    set_current_chat_id(chat_id)
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
                    ingest_text(text_content, source=filename, type="document")
                    prompt = caption if caption else "Procesá este documento y decime de qué trata"
                    full_message = f"{prompt}\n\n[Contenido de '{filename}']:\n{text_content[:6000]}"
                    set_current_chat_id(chat_id)
                    response = process_message(chat_id, full_message)
                    await send_message(chat_id, response)
                except Exception as e:
                    logger.error(f"Error en documento: {e}", exc_info=True)
                    await send_message(chat_id, "⚠️ Error procesando el documento.")
            background_tasks.add_task(handle_document)
            return {"ok": True}

        # Imagen / Screenshot
        photos = message.get("photo", [])
        if photos:
            async def handle_photo():
                await send_message(chat_id, "📸 Analizando imagen...")
                try:
                    import base64
                    from anthropic import Anthropic
                    best_photo = max(photos, key=lambda p: p.get("file_size", 0))
                    image_bytes = await download_telegram_file(best_photo["file_id"])
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                    # Analizar con Claude vision
                    anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                    prompt = caption if caption else "Analizá esta imagen. Si es un board de Azure DevOps, describí el estado de los features y OKRs. Si es otro tipo de imagen, describí su contenido relevante para el contexto de CTO."
                    response = anthropic_client.messages.create(
                        model="claude-opus-4-5",
                        max_tokens=1024,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                                {"type": "text", "text": prompt}
                            ]
                        }]
                    )
                    analysis = response.content[0].text

                    # Ingestar análisis en knowledge base
                    ingest_text(analysis, source="azure_screenshot", type="image")

                    # Procesar con el agente
                    full_message = f"[Análisis de imagen/screenshot]:\n{analysis}"
                    set_current_chat_id(chat_id)
                    agent_response = process_message(chat_id, full_message)
                    await send_message(chat_id, agent_response)
                except Exception as e:
                    logger.error(f"Error en imagen: {e}", exc_info=True)
                    await send_message(chat_id, "⚠️ Error procesando la imagen.")
            background_tasks.add_task(handle_photo)
            return {"ok": True}

        # Texto
        if text:
            async def handle_text():
                if len(text) > 200:
                    ingest_text(text, source="telegram", type="text")
                set_current_chat_id(chat_id)
                response = process_message(chat_id, text)
                await send_message(chat_id, response)
            background_tasks.add_task(handle_text)
            return {"ok": True}

        return {"ok": True}

    except Exception as e:
        logger.error(f"Error en webhook: {e}", exc_info=True)
        return {"ok": True}