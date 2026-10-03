import os
import httpx
from langchain_core.tools import tool

GRANOLA_API_KEY = os.environ.get("GRANOLA_API_KEY", "")
GRANOLA_BASE = "https://public-api.granola.ai/v1"

def _granola_get(path: str, params: dict = {}) -> dict:
    with httpx.Client(timeout=15) as client:
        r = client.get(
            f"{GRANOLA_BASE}{path}",
            headers={"Authorization": f"Bearer {GRANOLA_API_KEY}"},
            params=params
        )
        r.raise_for_status()
        return r.json()

@tool
def list_granola_meetings(limit: int = 5) -> str:
    """
    Lista las últimas reuniones de Granola con título y fecha.
    Usá cuando alguien pregunta por sus últimas reuniones o notas de Granola.
    """
    if not GRANOLA_API_KEY:
        return "⚠️ Granola no está configurado."
    try:
        data = _granola_get("/notes", {"page_size": limit})
        notes = data.get("notes", [])
        if not notes:
            return "No hay reuniones en Granola todavía."
        lines = ["📋 Últimas reuniones en Granola:"]
        for n in notes:
            title = n.get("title") or "Sin título"
            date = (n.get("created_at") or "")[:10]
            nid = n.get("id", "")
            lines.append(f"• {date} — {title} (id: {nid})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error al consultar Granola: {str(e)}"

@tool
def get_granola_meeting(note_id: str) -> str:
    """
    Trae el contenido completo de una reunión de Granola por su ID.
    Usá cuando alguien quiere ver los detalles o notas de una reunión específica.
    """
    if not GRANOLA_API_KEY:
        return "⚠️ Granola no está configurado."
    try:
        # Intentar con transcript primero, si falla sin él
        try:
            note = _granola_get(f"/notes/{note_id}", {"include": "transcript"})
        except httpx.HTTPStatusError:
            note = _granola_get(f"/notes/{note_id}")
        title = note.get("title") or "Sin título"
        date = (note.get("created_at") or "")[:10]
        summary = note.get("summary_markdown") or note.get("summary_text") or "Sin resumen."
        attendees = [a.get("name") or a.get("email", "") for a in note.get("attendees", [])]

        lines = [f"📝 {title} — {date}"]
        if attendees:
            lines.append(f"Participantes: {', '.join(attendees)}")
        lines.append(f"\n{summary}")
        return "\n".join(lines)
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (400, 404):
            return f"⚠️ La nota no tiene resumen generado por Granola todavía, o no se procesó completamente. Probá con otra reunión."
        return f"Error al obtener la nota: {str(e)}"
    except Exception as e:
        return f"Error al obtener la nota: {str(e)}"