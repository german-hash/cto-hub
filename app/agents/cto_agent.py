import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from app.tools.memory import save_memory, get_memory
from app.tools.crud import ALL_TOOLS

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
- Cuando alguien menciona una reunión o 1:1, ofrecé registrarla con create_one_on_one
- Cuando detectás un pendiente o tarea, ofrecé registrarla con create_task
- Cuando alguien pide sus pendientes, usá get_tasks
- Cuando alguien pide ver 1:1s, usá get_one_on_ones
- Si detectás un hecho importante, guardalo con save_memory
- Para updates a stakeholders usá lenguaje ejecutivo sin tecnicismos
- Cuando respondas por Telegram, usá formato simple sin markdown complejo

== MEMORIA PERSISTENTE ==
{memory}
"""

TOOLS = [save_memory, get_memory] + ALL_TOOLS

llm = ChatAnthropic(
    model="claude-opus-4-5",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
).bind_tools(TOOLS)

TOOL_MAP = {t.name: t for t in TOOLS}

def run_cto_agent(messages: list[dict], memory: str = "") -> str:
    """Ejecuta el CTO Agent con el historial de mensajes."""
    system = SystemMessage(content=SYSTEM_PROMPT.format(memory=memory or "Sin memoria cargada."))
    lc_messages = [system]

    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"] if isinstance(m["content"], str) else ""))

    # Agentic loop — máximo 5 iteraciones
    for _ in range(5):
        response = llm.invoke(lc_messages)

        if not response.tool_calls:
            return response.content

        # Procesar tool calls
        lc_messages.append(response)
        for tc in response.tool_calls:
            tool = TOOL_MAP.get(tc["name"])
            if tool:
                result = tool.invoke(tc["args"])
            else:
                result = f"Tool '{tc['name']}' no reconocida"
            lc_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return response.content
