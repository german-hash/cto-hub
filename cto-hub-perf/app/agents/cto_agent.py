import os
import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from app.tools.memory import save_memory, get_memory
from app.tools.crud import ALL_TOOLS
from app.tools.rag import search_knowledge, ingest_to_knowledge_base
from app.tools.calendar_tools import get_agenda, get_today_agenda
from app.tools.granola import list_granola_meetings, get_granola_meeting
from app.tools.usage import log_usage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Sos el asistente personal de German Guerriero, CTO de Tecnología Digital en Arcos Dorados (McDonald's Argentina).

Tu rol es ayudarlo a gestionar su equipo, proyectos, decisiones técnicas y comunicación con stakeholders.

== EQUIPO DIRECTO ==
- Pablo C: Chapter Lead BAs
- Her: Líder Mobile
- Gonza: Líder de Soporte
- Diego: DevOps
- Pablo N: Líder Arquitectura e Infra
- Caro: Líder QA
- Zorro: Líder iOS
- Saez: Líder Android
- Gallo: PO Flex Digital y Menu Editor (tiene a Alex y Pili como reportes)

== STAKEHOLDERS ==
- Diego M: VP Regional de Tecnología (jefe directo)
- Pablo E: Líder de Negocio Plataformas Digitales
- Carly: Líder de Negocio Digital

== ÁREAS ==
- Plataformas Digitales: app mobile propia de ecommerce QSR
- Flex Digital: hub de pedidos — expone catálogos e inserta pedidos en POS
- Menu Editor: organiza catálogos para Flex Digital

== CÓMO OPERAR ==
- Respondé en español rioplatense, directo y práctico
- Cuando alguien pregunta sobre agenda, reuniones o calendario, usá get_agenda o get_today_agenda
- Cuando alguien pregunta por reuniones de Granola o notas recientes, usá list_granola_meetings y get_granola_meeting
- Usá search_knowledge cuando la pregunta dependa de contexto histórico (reuniones, notas, documentos, decisiones pasadas). No lo uses para saludos, agenda, tareas o preguntas generales
- Cuando alguien menciona una reunión o 1:1, ofrecé registrarla con create_one_on_one
- Cuando detectás un pendiente o tarea, ofrecé registrarla con create_task
- Cuando alguien pide sus pendientes, usá get_tasks
- Cuando alguien pide ver 1:1s, usá get_one_on_ones
- Cuando alguien pide "guardá esto", "tomá nota de", usá ingest_to_knowledge_base
- Si detectás un hecho importante, guardalo con save_memory
- Para updates a stakeholders usá lenguaje ejecutivo sin tecnicismos
- Cuando respondas por Telegram, usá formato simple sin markdown complejo
- NUNCA empieces la respuesta con el nombre del usuario ni con ningún prefijo como "German:" o "Ger:". Respondé directamente.
- Sé conciso: respuestas cortas salvo que te pidan detalle
"""

MODEL = os.environ.get("CTO_AGENT_MODEL", "claude-sonnet-5-5")
MAX_MSG_CHARS = 2000      # tope por mensaje del historial (documentos largos no se re-envían enteros)
MAX_TOOL_RESULT_CHARS = 6000
CACHE = {"type": "ephemeral"}

TOOLS = [save_memory, get_memory, search_knowledge, ingest_to_knowledge_base,
         get_agenda, get_today_agenda,
         list_granola_meetings, get_granola_meeting] + ALL_TOOLS
TOOL_MAP = {t.name: t for t in TOOLS}

llm = ChatAnthropic(
    model=MODEL,
    api_key=os.environ.get("ANTHROPIC_API_KEY")
).bind_tools(TOOLS)


def _trim(text: str, limit: int = MAX_MSG_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + " …[recortado]"

def run_cto_agent(messages: list[dict], memory: str = "") -> str:
    # Prompt caching: tools + system estático + memoria quedan cacheados entre llamadas
    # (las lecturas de caché cuestan ~10% del precio normal de input).
    system = SystemMessage(content=[
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": CACHE},
        {"type": "text", "text": "== MEMORIA PERSISTENTE ==\n" + (memory or "Sin memoria cargada."),
         "cache_control": CACHE},
    ])
    lc_messages = [system]

    # Filtrar mensajes válidos y asegurar que termine en usuario
    valid_messages = []
    for m in messages:
        if m["role"] == "user":
            # Ignorar tool results del historial (son JSON)
            c = m["content"]
            if isinstance(c, str) and c.strip() and not c.strip().startswith('[{'):
                valid_messages.append(HumanMessage(content=_trim(c)))
        elif m["role"] == "assistant":
            c = m["content"]
            if isinstance(c, str) and c.strip():
                valid_messages.append(AIMessage(content=_trim(c)))

    # Asegurar que el historial no termine con assistant
    while valid_messages and isinstance(valid_messages[-1], AIMessage):
        valid_messages.pop()

    # Cache breakpoint en el último mensaje del usuario: en cada vuelta del loop de tools
    # el historial completo se lee de caché en vez de pagarse entero de nuevo.
    if valid_messages and isinstance(valid_messages[-1], HumanMessage):
        last = valid_messages[-1]
        valid_messages[-1] = HumanMessage(content=[
            {"type": "text", "text": last.content, "cache_control": CACHE}
        ])

    lc_messages.extend(valid_messages)

    for i in range(10):
        response = llm.invoke(lc_messages)
        log_usage("cto_agent", MODEL, getattr(response, "usage_metadata", None))
        logger.info(f"Iteración {i} — tool_calls: {len(response.tool_calls)}, content len: {len(str(response.content))}")

        if not response.tool_calls:
            # Filtrar bloques thinking y extraer solo el texto
            raw = response.content
            if isinstance(raw, list):
                text = " ".join(
                    b.get("text", "") if isinstance(b, dict) else (b.text if hasattr(b, "text") else "")
                    for b in raw
                    if (isinstance(b, dict) and b.get("type") == "text") or
                       (hasattr(b, "type") and b.type == "text")
                ).strip()
            else:
                text = str(raw).strip()
            if not text:
                logger.warning("Respuesta vacía del agente")
                return "No pude generar una respuesta. Intentá de nuevo."
            return text

        lc_messages.append(response)
        for tc in response.tool_calls:
            logger.info(f"Tool call: {tc['name']} — args: {tc['args']}")
            tool = TOOL_MAP.get(tc["name"])
            if tool:
                result = tool.invoke(tc["args"])
                logger.info(f"Tool result: {str(result)[:200]}")
            else:
                result = f"Tool '{tc['name']}' no reconocida"
            lc_messages.append(ToolMessage(content=_trim(str(result), MAX_TOOL_RESULT_CHARS), tool_call_id=tc["id"]))

    return "Alcancé el límite de iteraciones. Intentá reformular la pregunta."