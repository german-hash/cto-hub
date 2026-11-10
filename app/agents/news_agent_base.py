import os
from datetime import date
from anthropic import Anthropic
from tavily import TavilyClient
from app.tools.usage import log_usage

NEWS_MODEL = "claude-haiku-4-5-20251001"
MAX_NEWS_ITERATIONS = 6  # antes era while True: un loop sin tope podía quemar tokens

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]


class NewsAgentError(RuntimeError):
    """El agente no produjo un resumen utilizable. El endpoint responde 502 y Make corta (no manda audio vacío)."""


def _fecha_hoy() -> str:
    hoy = date.today()
    return f"{hoy.day} de {MESES[hoy.month - 1]} de {hoy.year}"


def _contexto_fecha() -> str:
    # Sin esto, el modelo (con conocimiento hasta antes de la fecha real) cree que la fecha pedida es "futura"
    # y responde que no puede buscar noticias en vez de usar la tool.
    return (f"\n\nCONTEXTO TEMPORAL: hoy es {_fecha_hoy()}. Esta fecha es real y actual, aunque sea posterior a "
            "tu entrenamiento. Nunca digas que la fecha es futura ni que no podés buscar: usá siempre la tool "
            "buscar_noticias y confiá en sus resultados. No hagas preguntas al usuario: entregá directamente el "
            "resumen en el formato pedido.")


def _texto_final(content) -> str:
    return "\n".join(b.text for b in content if getattr(b, "type", None) == "text" and b.text).strip()


def run_news_agent(topic: str, system_prompt: str) -> str:
    """Función base para todos los agentes de noticias. Lanza NewsAgentError si no hay un resumen válido."""
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

    tools = [{
        "name": "buscar_noticias",
        "description": f"Busca noticias recientes de {topic} en internet (últimos 14 días)",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "El término de búsqueda"}
            },
            "required": ["query"]
        }
    }]

    messages = [{"role": "user", "content": f"Cuales son las noticias de {topic} de la semana al {_fecha_hoy()}?"}]
    busquedas = 0

    for _ in range(MAX_NEWS_ITERATIONS):
        response = client.messages.create(
            model=NEWS_MODEL,
            max_tokens=2048,
            system=system_prompt + _contexto_fecha(),
            tools=tools,
            messages=messages
        )
        log_usage(f"news:{topic}", NEWS_MODEL, response.usage)

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    busquedas += 1
                    resultado = tavily.search(
                        query=block.input["query"],
                        topic="news",            # 'days' solo filtra por fecha con topic="news"
                        search_depth="advanced",
                        max_results=5,
                        include_raw_content=False,
                        days=14
                    )
                    noticias_texto = ""
                    for r in resultado.get("results", []):
                        fecha = r.get("published_date") or ""
                        noticias_texto += (f"- Titulo: {r['title']}\n  Fecha: {fecha}\n  Resumen: {r['content'][:300]}\n"
                                           f"  Fuente: {r['url']}\n\n")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": noticias_texto or "Sin resultados para esta búsqueda; probá con otros términos."
                    })
            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason in ("end_turn", "max_tokens"):
            texto = _texto_final(response.content)
            if busquedas == 0 or "FUENTES_INICIO" not in texto or len(texto) < 500:
                raise NewsAgentError(f"respuesta inválida para {topic} (búsquedas={busquedas}): {texto[:200]!r}")
            return texto
        else:
            break

    raise NewsAgentError(f"sin resumen para {topic} tras {MAX_NEWS_ITERATIONS} iteraciones")


def run_tech_news_agent() -> dict:
    texto = run_news_agent("tecnologia", """Eres un agente especializado en noticias de tecnologia.
Cuando el usuario te pida noticias, usas la tool buscar_noticias para buscar informacion actualizada.
IMPORTANTE:
- Priorizas noticias de los ultimos 14 dias, pero podés incluir hasta 30 dias si hay pocas recientes
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
- Priorizas noticias de los ultimos 14 dias, pero podés incluir hasta 30 dias si hay pocas recientes
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


def run_finance_news_agent() -> dict:
    texto = run_news_agent("finanzas y global macro", """Eres un agente especializado en noticias de finanzas y global macro.
Cuando el usuario te pida noticias, usas la tool buscar_noticias para buscar informacion actualizada.
IMPORTANTE:
- Priorizas noticias de los ultimos 14 dias, pero podés incluir hasta 30 dias si hay pocas recientes
- Presentas entre 8 y 10 noticias, entre 5 y 6 del mundo y entre 3 y 4 de latam
- El texto debe estar escrito para ser LEIDO EN VOZ ALTA, sin markdown
- No uses simbolos como #, *, **, ---, emojis ni caracteres especiales
- Escribi en texto plano corrido, como un locutor de radio
- Cada noticia debe tener: titulo, resumen de 2 lineas y fuente
- Separas cada noticia con un punto y aparte
- El texto total no debe superar los 3500 caracteres
- Temas prioritarios: tasas de interes, inflacion, mercados de valores, politica economica, commodities
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