import os
import logging
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from pathlib import Path
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
from app.agents.news_agent_base import run_tech_news_agent, run_qsr_news_agent, run_finance_news_agent
from app.agents.stock_screener_agent import run_stock_screener_agent
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
/dashboard — Link al dashboard web
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
    """Envía mensaje a Telegram dividiendo si supera el límite de 4096 chars."""
    MAX_LEN = 4000
    chunks = [text[i:i+MAX_LEN] for i in range(0, len(text), MAX_LEN)]
    async with httpx.AsyncClient() as client:
        for chunk in chunks:
            r = await client.post(f"{TELEGRAM_API}/sendMessage", json={
                "chat_id": chat_id,
                "text": chunk
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

@app.get("/agents/tech-news")
async def tech_news_endpoint():
    """Endpoint GET para Make — devuelve noticias tech + fuentes para TTS."""
    result = run_tech_news_agent()
    return result

@app.get("/agents/qsr-news")
async def qsr_news_endpoint():
    """Endpoint GET para Make — devuelve noticias QSR + fuentes para TTS."""
    result = run_qsr_news_agent()
    return result

@app.get("/agents/finance-news")
async def finance_news_endpoint():
    """Endpoint GET para Make — devuelve noticias finanzas + fuentes para TTS."""
    result = run_finance_news_agent()
    return result

@app.post("/agents/stock-screener")
async def stock_screener_endpoint(request: Request, background_tasks: BackgroundTasks):
    """Endpoint para Make — ejecuta el screener de acciones y manda el resultado a Telegram."""
    data = await request.json()
    chat_id = str(data.get("chat_id", ""))
    if not chat_id:
        return {"error": "chat_id requerido"}

    async def handle():
        await send_message(chat_id, "📈 Analizando acciones... puede tardar 1-2 minutos.")
        result = run_stock_screener_agent()
        await send_message(chat_id, result)

    background_tasks.add_task(handle)
    return {"ok": True}

# ── Granola Webhook ──────────────────────────────────────────────────────────

@app.post("/webhooks/granola")
async def granola_webhook(request: Request, background_tasks: BackgroundTasks):
    """Recibe eventos de Granola, busca el contenido completo via API y lo ingesta."""
    try:
        data = await request.json()
        logger.info(f"Granola webhook: {data}")

        async def handle():
            import httpx
            from app.tools.rag import ingest_text

            GRANOLA_API_KEY = os.environ.get("GRANOLA_API_KEY", "")
            if not GRANOLA_API_KEY:
                logger.error("GRANOLA_API_KEY no configurada")
                return

            # Obtener el note_id del evento
            note_id = (data.get("note_id") or data.get("id") or
                      data.get("data", {}).get("note_id") or
                      data.get("data", {}).get("id"))

            if not note_id:
                logger.warning(f"Granola webhook: no encontré note_id en {data}")
                return

            # Buscar el contenido completo via API de Granola
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    f"https://api.granola.ai/v1/notes/{note_id}",
                    headers={"Authorization": f"Bearer {GRANOLA_API_KEY}"}
                )
                if r.status_code != 200:
                    logger.error(f"Granola API error: {r.status_code} {r.text[:200]}")
                    return
                note = r.json()

            title = note.get("title", "Reunión sin título")
            notes = note.get("notes", "") or note.get("summary", "") or note.get("content", "")
            transcript = note.get("transcript", "")
            created_at = note.get("created_at", "") or note.get("date", "")

            if not notes and not transcript:
                logger.warning(f"Granola: nota '{title}' vacía, ignorando")
                return

            text_parts = [f"Reunión: {title}"]
            if created_at:
                text_parts.append(f"Fecha: {created_at}")
            if notes:
                text_parts.append(f"Notas:\n{notes}")
            if transcript:
                text_parts.append(f"Transcripción:\n{transcript[:3000]}")

            full_text = "\n\n".join(text_parts)
            chunks = ingest_text(full_text, source=f"granola/{title}", type="meeting")
            logger.info(f"Granola: ingresté '{title}' — {chunks} chunks")

        background_tasks.add_task(handle)
        return {"ok": True}

    except Exception as e:
        logger.error(f"Error en Granola webhook: {e}", exc_info=True)
        return {"ok": True}

# ── Dashboard ────────────────────────────────────────────────────────────────

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Sirve el dashboard HTML."""
    html = Path("dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)

@app.get("/dashboard/data/tasks")
async def dashboard_tasks():
    from app.tools.supabase_client import supabase, _retry
    def _fn():
        return supabase.table("tasks").select("title,priority,related_person").eq("status", "pending").order("created_at", desc=True).limit(10).execute()
    result = _retry(_fn)
    return result.data

@app.get("/dashboard/data/memory")
async def dashboard_memory():
    from app.tools.supabase_client import supabase, _retry
    def _fn():
        return supabase.table("cto_memory").select("category,content").order("created_at", desc=False).limit(20).execute()
    result = _retry(_fn)
    return result.data

@app.get("/dashboard/data/oneononees")
async def dashboard_oneononees():
    from app.tools.supabase_client import supabase, _retry
    def _fn():
        return supabase.table("one_on_ones").select("person,date,topics").order("date", desc=True).limit(6).execute()
    result = _retry(_fn)
    return result.data

@app.post("/dashboard/ask")
async def dashboard_ask(request: Request):
    data = await request.json()
    message = data.get("message", "").strip()
    if not message:
        return {"response": "Mensaje vacío"}
    from app.graph import process_message
    response = process_message("dashboard", message)
    return {"response": response}

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

        if text.lower() == "/dashboard":
            await send_message(chat_id, "📊 Dashboard: https://cto-hub.onrender.com/dashboard")
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