import os
import httpx
import base64

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.ogg") -> str:
    """Transcribe audio usando OpenAI Whisper."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": (filename, audio_bytes, "audio/ogg")},
            data={"model": "whisper-1", "language": "es"}
        )
        r.raise_for_status()
        return r.json().get("text", "")

async def download_telegram_file(file_id: str) -> bytes:
    """Descarga un archivo de Telegram y devuelve los bytes."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id})
        r.raise_for_status()
        file_path = r.json()["result"]["file_path"]
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        r2 = await client.get(file_url)
        r2.raise_for_status()
        return r2.content

async def extract_text_from_document(file_bytes: bytes, mime_type: str, filename: str) -> str:
    """Extrae texto de un documento según su tipo."""
    # Texto plano
    if mime_type in ("text/plain",) or filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    # PDF — usar Anthropic vision
    if mime_type == "application/pdf" or filename.endswith(".pdf"):
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "base64", "media_type": "application/pdf", "data": b64}
                    },
                    {"type": "text", "text": "Extraé todo el texto de este documento tal cual está, sin resumir."}
                ]
            }]
        )
        return response.content[0].text

    # Otros formatos — devolver como texto si es posible
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return "No pude extraer texto de este tipo de archivo."
