import logging
import httpx
from datetime import datetime, timedelta
from langchain_core.tools import tool
from app.tools.calendar import get_tokens, save_tokens, format_events, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET

logger = logging.getLogger(__name__)

_current_chat_id = ""

def set_current_chat_id(chat_id: str):
    global _current_chat_id
    _current_chat_id = chat_id

def _refresh_token_sync(refresh_tok: str) -> str:
    r = httpx.post("https://oauth2.googleapis.com/token", data={
        "refresh_token": refresh_tok,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "grant_type": "refresh_token"
    })
    r.raise_for_status()
    return r.json()["access_token"]

def _get_events_sync(chat_id: str, days: int) -> list[dict]:
    """Versión sincrónica de get_calendar_events."""
    tokens = get_tokens(chat_id)
    if not tokens:
        return []

    access_token = tokens["access_token"]
    refresh_tok = tokens.get("refresh_token", "")

    now = datetime.utcnow()
    params = {
        "timeMin": now.isoformat() + "Z",
        "timeMax": (now + timedelta(days=days)).isoformat() + "Z",
        "singleEvents": True,
        "orderBy": "startTime",
        "maxResults": 20
    }

    with httpx.Client(timeout=15) as client:
        r = client.get(
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            headers={"Authorization": f"Bearer {access_token}"},
            params=params
        )
        if r.status_code == 401 and refresh_tok:
            access_token = _refresh_token_sync(refresh_tok)
            save_tokens(chat_id, {"access_token": access_token, "refresh_token": refresh_tok})
            r = client.get(
                "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                headers={"Authorization": f"Bearer {access_token}"},
                params=params
            )
        r.raise_for_status()
        return r.json().get("items", [])

@tool
def get_agenda(days: int = 7) -> str:
    """
    Trae los eventos del Google Calendar de los próximos días.
    Usá cuando alguien pregunta por su agenda, reuniones o calendario.
    """
    tokens = get_tokens(_current_chat_id)
    if not tokens:
        return "⚠️ Google Calendar no está conectado. Mandá /conectar_calendar para autorizarlo."
    try:
        events = _get_events_sync(_current_chat_id, days)
        return format_events(events)
    except Exception as e:
        logger.error(f"Error leyendo calendario: {e}", exc_info=True)
        return f"Error al leer el calendario: {str(e)}"

@tool
def get_today_agenda() -> str:
    """Trae los eventos de hoy del Google Calendar."""
    return get_agenda.invoke({"days": 1})
