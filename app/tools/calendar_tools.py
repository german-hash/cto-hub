import asyncio
import logging
from langchain_core.tools import tool
from app.tools.calendar import get_calendar_events, format_events, get_access_token

logger = logging.getLogger(__name__)

_current_chat_id = ""

def set_current_chat_id(chat_id: str):
    global _current_chat_id
    _current_chat_id = chat_id

@tool
def get_agenda(days: int = 7) -> str:
    """
    Trae los eventos del Google Calendar de los próximos días.
    Usá cuando alguien pregunta por su agenda, reuniones o calendario.
    """
    if not get_access_token(_current_chat_id):
        return "⚠️ Google Calendar no está conectado. Mandá /conectar_calendar para autorizarlo."
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        events = loop.run_until_complete(get_calendar_events(_current_chat_id, days))
        loop.close()
        return format_events(events)
    except Exception as e:
        logger.error(f"Error leyendo calendario: {e}", exc_info=True)
        return f"Error al leer el calendario: {str(e)}"

@tool
def get_today_agenda() -> str:
    """Trae los eventos de hoy del Google Calendar."""
    return get_agenda.invoke({"days": 1})
