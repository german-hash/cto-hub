import os
from openai import OpenAI
from app.tools.supabase_client import supabase, _retry
from langchain_core.tools import tool

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def get_embedding(text: str) -> list[float]:
    """Genera embedding con OpenAI text-embedding-3-small."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000]
    )
    return response.data[0].embedding

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide texto en chunks con overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def ingest_text(content: str, source: str = "", type: str = "text") -> int:
    """
    Ingesta texto en la knowledge base.
    Chunkea, genera embeddings y guarda en Supabase.
    Devuelve la cantidad de chunks guardados.
    """
    chunks = chunk_text(content)
    saved = 0
    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            embedding = get_embedding(chunk)
            def _fn(c=chunk, e=embedding):
                supabase.table("knowledge_chunks").insert({
                    "content": c,
                    "embedding": e,
                    "source": source,
                    "type": type
                }).execute()
            _retry(_fn)
            saved += 1
        except Exception as e:
            print(f"Error al guardar chunk: {e}")
    return saved

@tool
def search_knowledge(query: str, limit: int = 5) -> str:
    """
    Busca semánticamente en la knowledge base del CTO.
    Usá esta tool cuando necesitás recordar algo que fue ingresado antes
    (notas, audios, documentos, URLs).
    """
    try:
        embedding = get_embedding(query)
        result = supabase.rpc("match_knowledge_chunks", {
            "query_embedding": embedding,
            "match_count": limit
        }).execute()

        if not result.data:
            return "No encontré nada relacionado en la knowledge base."

        lines = [f"Resultados para '{query}':"]
        for r in result.data:
            source = f" [{r['source']}]" if r.get("source") else ""
            lines.append(f"\n---{source}")
            lines.append(r["content"])

        return "\n".join(lines)
    except Exception as e:
        return f"Error en búsqueda: {str(e)}"

@tool
def ingest_to_knowledge_base(content: str, source: str = "", type: str = "text") -> str:
    """
    Ingesta texto en la knowledge base para búsqueda futura.
    Usá cuando el usuario pide guardar información para recordarla después.
    """
    try:
        saved = ingest_text(content, source, type)
        return f"✅ Guardé {saved} chunks en la knowledge base ({source or 'sin fuente'})."
    except Exception as e:
        return f"Error al ingestar: {str(e)}"
