import os
import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from app.tools.memory import save_memory, get_memory
from app.tools.crud import ALL_TOOLS
from app.tools.rag import search_knowledge, ingest_to_knowledge_base
from app.tools.calendar_tools import get_agenda, get_today_agenda
from app.tools.granola import list_granola_meetings, get_granola_meeting

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
- SIEMPRE que alguien pregunte sobre un tema específico, usá search_knowledge primero
- Cuando alguien menciona una reunión o 1:1, ofrecé registrarla con create_one_on_one
- Cuando detectás un pendiente o tarea, ofrecé registrarla con create_task
- Cuando alguien pide sus pendientes, usá get_tasks
- Cuando alguien pide ver 1:1s, usá get_one_on_ones
- Cuando alguien pide "guardá esto", "tomá nota de", usá ingest_to_knowledge_base
- Si detectás un hecho importante, guardalo con save_memory
- Para updates a stakeholders usá lenguaje ejecutivo sin tecnicismos
- Cuando respondas por Telegram, usá formato simple sin markdown complejo

== MEMORIA PERSISTENTE ==
{memory}
"""

TOOLS = [save_memory, get_memory, search_knowledge, ingest_to_knowledge_base,
         get_agenda, get_today_agenda,
         list_granola_meetings, get_granola_meeting] + ALL_TOOLS
TOOL_MAP = {t.name: t for t in TOOLS}

llm = ChatAnthropic(
    model="claude-sonnet-5-5",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
).bind_tools(TOOLS)

def run_cto_agent(messages: list[dict], memory: str = "") -> str:
    system = SystemMessage(content=SYSTEM_PROMPT.format(memory=memory or "Sin memoria cargada."))
    lc_messages = [system]

    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            content = m["content"]
            if isinstance(content, str) and content.strip():
                lc_messages.append(AIMessage(content=content))

    for i in range(10):
        response = llm.invoke(lc_messages)
        logger.info(f"Iteración {i} — tool_calls: {len(response.tool_calls)}, content len: {len(str(response.content))}")

        if not response.tool_calls:
            text = response.content
            if not text or not str(text).strip():
                logger.warning("Respuesta vacía del agente")
                return "No pude generar una respuesta. Intentá de nuevo."
            return str(text)

        lc_messages.append(response)
        for tc in response.tool_calls:
            logger.info(f"Tool call: {tc['name']} — args: {tc['args']}")
            tool = TOOL_MAP.get(tc["name"])
            if tool:
                result = tool.invoke(tc["args"])
                logger.info(f"Tool result: {str(result)[:200]}")
            else:
                result = f"Tool '{tc['name']}' no reconocida"
            lc_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return "Alcancé el límite de iteraciones. Intentá reformular la pregunta."
