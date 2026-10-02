from langchain_core.tools import tool

# chat_id global para las tools (se setea antes de llamar al agente)
_current_chat_id = ""

def set_current_chat_id(chat_id: str):
    global _current_chat_id
    _current_chat_id = chat_id

@tool
def get_agenda(days: int = 7) -> str:
    """
    Trae los eventos del Google Calendar de los próximos días.
    Usá esta tool cuando alguien pregunta por su agenda, reuniones o calendario.
    """
    import asyncio
    from app.tools.calendar import get_calendar_events, format_events, get_access_token

    if not get_access_token(_current_chat_id):
        return "⚠️ No está conectado Google Calendar. Mandá /conectar_calendar para autorizarlo."

    try:
        events = asyncio.run(get_calendar_events(_current_chat_id, days))
        return format_events(events)
    except Exception as e:
        return f"Error al leer el calendario: {str(e)}"

@tool
def get_today_agenda() -> str:
    """Trae los eventos de hoy del Google Calendar."""
    return get_agenda.invoke({"days": 1})
