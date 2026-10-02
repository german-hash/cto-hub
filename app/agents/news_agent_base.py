import os
from datetime import date
from anthropic import Anthropic
from tavily import TavilyClient

def run_news_agent(topic: str, system_prompt: str) -> str:
    """Función base para todos los agentes de noticias."""
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

    tools = [{
        "name": "buscar_noticias",
        "description": f"Busca noticias recientes sobre {topic} en internet",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Término de búsqueda"}
            },
            "required": ["query"]
        }
    }]

    hoy = date.today().strftime("%d %B %Y")
    messages = [{"role": "user", "content": f"¿Cuáles son las noticias de {topic} del día de hoy {hoy}?"}]

    for _ in range(5):
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            system=system_prompt,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    resultado = tavily.search(
                        query=block.input["query"],
                        search_depth="advanced",
                        max_results=5,
                        include_raw_content=False,
                        days=1
                    )
                    noticias_texto = ""
                    for r in resultado["results"]:
                        noticias_texto += f"- Título: {r['title']}\n  Resumen: {r['content'][:300]}\n  Fuente: {r['url']}\n\n"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": noticias_texto
                    })
            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

    return f"No pude obtener las noticias de {topic} del día."


def run_tech_news_agent() -> str:
    hoy = date.today().strftime("%d/%m/%Y")
    return run_news_agent("tecnología", f"""Eres un agente especializado en noticias de tecnología.
Cuando el usuario te pida noticias, usás la tool buscar_noticias para buscar información actualizada.
IMPORTANTE:
- Solo mostrás noticias del día de hoy ({hoy})
- Presentás MÍNIMO 5 noticias, idealmente 8 o más
- Cada noticia debe tener: titular, resumen de 2-3 líneas y fuente
- Organizalas por categorías: IA, Gadgets, Startups, Software, etc.
- Respondés siempre en español.""")


def run_qsr_news_agent() -> str:
    hoy = date.today().strftime("%d/%m/%Y")
    return run_news_agent("QSR (Quick Service Restaurants)", f"""Eres un agente especializado en noticias de QSR.
Cuando el usuario te pida noticias, usás la tool buscar_noticias para buscar información actualizada.
IMPORTANTE:
- Solo mostrás noticias del día de hoy ({hoy})
- Presentás entre 5 y 8 noticias, priorizando noticias de Latam
- Cada noticia debe tener: titular, resumen de 2-3 líneas y fuente
- Organizalas por categorías: IA, Ecommerce, Tecnología, Tendencias, etc.
- Respondés siempre en español.""")


def run_finance_news_agent() -> str:
    hoy = date.today().strftime("%d/%m/%Y")
    return run_news_agent("finanzas y global macro", f"""Eres un agente especializado en noticias de finanzas y global macro.
Cuando el usuario te pida noticias, usás la tool buscar_noticias para buscar información actualizada.
IMPORTANTE:
- Solo mostrás noticias del día de hoy ({hoy})
- Presentás entre 5 y 8 noticias, priorizando noticias de Latam
- Cada noticia debe tener: titular, resumen de 2-3 líneas y fuente
- Organizalas por categorías: económicas, políticas, mercados de valores, etc.
- Respondés siempre en español.""")
