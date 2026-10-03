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
        "description": f"Busca noticias recientes de {topic} en internet",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "El término de búsqueda"}
            },
            "required": ["query"]
        }
    }]

    hoy_query = date.today().strftime("%d %B %Y")
    messages = [{"role": "user", "content": f"Cuales son las noticias de {topic} de la semana {hoy_query}?"}]

    while True:
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
                        days=7
                    )
                    noticias_texto = ""
                    for r in resultado["results"]:
                        noticias_texto += f"- Titulo: {r['title']}\n  Resumen: {r['content'][:300]}\n  Fuente: {r['url']}\n\n"
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
            break

    return "No pude obtener las noticias."


def run_tech_news_agent() -> dict:
    texto = run_news_agent("tecnologia", """Eres un agente especializado en noticias de tecnologia.
Cuando el usuario te pida noticias, usas la tool buscar_noticias para buscar informacion actualizada.
IMPORTANTE:
- Solo mostras noticias de los ultimos 7 dias
- Presentas entre 8 y 10 noticias, entre 5 y 6 del mundo y entre 3 y 4 de latam
- El texto debe estar escrito para ser LEIDO EN VOZ ALTA, sin markdown
- No uses simbolos como #, *, **, ---, emojis ni caracteres especiales
- Escribi en texto plano corrido, como un locutor de radio
- Cada noticia debe tener: titulo, resumen de 2 lineas y fuente
- Separas cada noticia con un punto y aparte
- El texto total no debe superar los 3500 caracteres
- Al final del texto agrega una seccion separada con el texto exacto:
  FUENTES_INICIO
  luego una linea por cada fuente en formato: Titulo - URL
  luego el texto exacto: FUENTES_FIN
- Respondes siempre en español""")

    if "FUENTES_INICIO" in texto and "FUENTES_FIN" in texto:
        partes = texto.split("FUENTES_INICIO")
        texto_audio = partes[0].strip()
        fuentes = partes[1].split("FUENTES_FIN")[0].strip()
    else:
        texto_audio = texto
        fuentes = "No se encontraron fuentes"

    return {"noticias": texto_audio, "fuentes": fuentes}


def run_qsr_news_agent() -> dict:
    texto = run_news_agent("QSR (Quick Service Restaurants)", """Eres un agente especializado en noticias de QSR.
Cuando el usuario te pida noticias, usas la tool buscar_noticias para buscar informacion actualizada.
IMPORTANTE:
- Solo mostras noticias de los ultimos 7 dias
- Presentas entre 8 y 10 noticias, entre 5 y 6 del mundo y entre 3 y 4 de latam
- El texto debe estar escrito para ser LEIDO EN VOZ ALTA, sin markdown
- No uses simbolos como #, *, **, ---, emojis ni caracteres especiales
- Escribi en texto plano corrido, como un locutor de radio
- Cada noticia debe tener: titulo, resumen de 2 lineas y fuente
- Separas cada noticia con un punto y aparte
- El texto total no debe superar los 3500 caracteres
- Priorizas noticias de digitalizacion y ecommerce
- Priorizas noticias de cadenas globales como McDonalds, Starbucks, Burger King, KFC, Subway, Pizza Hut, Dominos, y cadenas relevantes en latinoamerica
- Al final del texto agrega una seccion separada con el texto exacto:
  FUENTES_INICIO
  luego una linea por cada fuente en formato: Titulo - URL
  luego el texto exacto: FUENTES_FIN
- Respondes siempre en español""")

    if "FUENTES_INICIO" in texto and "FUENTES_FIN" in texto:
        partes = texto.split("FUENTES_INICIO")
        texto_audio = partes[0].strip()
        fuentes = partes[1].split("FUENTES_FIN")[0].strip()
    else:
        texto_audio = texto
        fuentes = "No se encontraron fuentes"

    return {"noticias": texto_audio, "fuentes": fuentes}


def run_finance_news_agent() -> str:
    return run_news_agent("finanzas y global macro", """Eres un agente especializado en noticias de finanzas y global macro.
Cuando el usuario te pida noticias, usas la tool buscar_noticias para buscar informacion actualizada.
IMPORTANTE:
- Solo mostras noticias de los ultimos 7 dias
- Presentas entre 8 y 10 noticias, entre 5 y 6 del mundo y entre 3 y 4 de latam
- El texto debe estar escrito para ser LEIDO EN VOZ ALTA, sin markdown
- No uses simbolos como #, *, **, ---, emojis ni caracteres especiales
- Escribi en texto plano corrido, como un locutor de radio
- Cada noticia debe tener: titulo, resumen de 2 lineas y fuente
- Separas cada noticia con un punto y aparte
- El texto total no debe superar los 3500 caracteres
- Respondes siempre en español""")
