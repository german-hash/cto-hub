import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.tools.memory import save_memory, get_memory

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
- Si detectás un hecho importante que Ger debería recordar, usá la tool save_memory
- Cuando necesitás contexto previo, usá la tool get_memory
- Para updates a stakeholders usá lenguaje ejecutivo sin tecnicismos

== MEMORIA ==
{memory}
"""

llm = ChatAnthropic(
    model="claude-opus-4-5",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
).bind_tools([save_memory, get_memory])

def run_cto_agent(messages: list[dict], memory: str = "") -> str:
    """Ejecuta el CTO Agent con el historial de mensajes."""
    system = SystemMessage(content=SYSTEM_PROMPT.format(memory=memory or "Sin memoria cargada."))
    lc_messages = [system]

    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    response = llm.invoke(lc_messages)

    # Procesar tool calls si las hay
    if response.tool_calls:
        from langchain_core.messages import ToolMessage
        tool_results = []
        for tc in response.tool_calls:
            if tc["name"] == "save_memory":
                result = save_memory.invoke(tc["args"])
            elif tc["name"] == "get_memory":
                result = get_memory.invoke(tc["args"])
            else:
                result = "Tool no reconocida"
            tool_results.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        lc_messages.append(response)
        lc_messages.extend(tool_results)
        final = llm.invoke(lc_messages)
        return final.content

    return response.content
