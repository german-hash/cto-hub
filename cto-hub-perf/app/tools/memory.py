from app.tools.supabase_client import supabase, _retry
from langchain_core.tools import tool

@tool
def save_memory(category: str, content: str) -> str:
    """Guarda un hecho importante en la memoria persistente del CTO."""
    def _fn():
        supabase.table("cto_memory").insert({
            "category": category.lower().strip(),
            "content": content.strip()
        }).execute()
    try:
        _retry(_fn)
        return f"✅ Guardado en memoria [{category}]: {content}"
    except Exception as e:
        return f"Error al guardar en memoria: {str(e)}"

@tool
def get_memory() -> str:
    """Lee toda la memoria persistente del CTO."""
    def _fn():
        result = supabase.table("cto_memory") \
            .select("category, content") \
            .order("created_at", desc=False) \
            .execute()
        if not result.data:
            return "No hay nada en memoria todavía."
        lines = [f"[{r['category'].upper()}] {r['content']}" for r in result.data]
        return "\n".join(lines)
    try:
        return _retry(_fn)
    except Exception:
        return ""

def get_history(chat_id: str, limit: int = 12) -> list[dict]:
    """Lee los ÚLTIMOS `limit` mensajes del historial (en orden cronológico)."""
    def _fn():
        result = supabase.table("conversation_history") \
            .select("role, content") \
            .eq("chat_id", chat_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        rows = list(reversed(result.data or []))
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    try:
        return _retry(_fn)
    except Exception:
        return []

def get_memory_for_prompt(max_chars: int = 3000) -> str:
    """Memoria acotada para el system prompt: las entradas más recientes hasta `max_chars`.
    La memoria completa sigue disponible con la tool get_memory."""
    def _fn():
        return supabase.table("cto_memory") \
            .select("category, content") \
            .order("created_at", desc=True) \
            .limit(200) \
            .execute()
    try:
        rows = _retry(_fn).data or []
    except Exception:
        return ""
    lines, total = [], 0
    for r in rows:
        line = f"[{r['category'].upper()}] {r['content']}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    lines.reverse()
    if len(lines) < len(rows):
        lines.append(f"(+{len(rows) - len(lines)} entradas más antiguas: usá get_memory si hacen falta)")
    return "\n".join(lines)

def save_message(chat_id: str, role: str, content: str):
    """Guarda un mensaje en el historial de conversación."""
    def _fn():
        supabase.table("conversation_history").insert({
            "chat_id": chat_id,
            "role": role,
            "content": content
        }).execute()
    try:
        _retry(_fn)
    except Exception:
        pass

def reset_history(chat_id: str):
    """Borra el historial de conversación de un chat."""
    def _fn():
        supabase.table("conversation_history") \
            .delete() \
            .eq("chat_id", chat_id) \
            .execute()
    try:
        _retry(_fn)
    except Exception:
        pass
