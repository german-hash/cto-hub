from datetime import date
from app.tools.supabase_client import supabase, _retry
from langchain_core.tools import tool

# ── ONE ON ONES ──────────────────────────────────────────────────────────────

@tool
def create_one_on_one(person: str, topics: list, action_items: list = [], notes: str = "") -> str:
    """Registra un 1:1 o reunión con alguien del equipo o stakeholder."""
    def _fn():
        supabase.table("one_on_ones").insert({
            "person": person.lower().strip(),
            "date": date.today().isoformat(),
            "topics": topics,
            "action_items": action_items,
            "notes": notes
        }).execute()
    try:
        _retry(_fn)
        topics_str = "\n".join([f"• {t}" for t in topics])
        return f"✅ 1:1 con {person} registrado ({date.today().strftime('%d/%m/%Y')}):\n{topics_str}"
    except Exception as e:
        return f"Error al registrar 1:1: {str(e)}"

@tool
def get_one_on_ones(person: str = "", limit: int = 5) -> str:
    """Trae los últimos 1:1s. Si se especifica una persona, filtra por ella."""
    def _fn():
        query = supabase.table("one_on_ones") \
            .select("person, date, topics, action_items, notes") \
            .order("date", desc=True) \
            .limit(limit)
        if person:
            query = query.eq("person", person.lower().strip())
        return query.execute()
    try:
        result = _retry(_fn)
        if not result.data:
            return f"No hay 1:1s registrados{' con ' + person if person else ''}."
        lines = []
        for r in result.data:
            lines.append(f"\n📅 {r['person'].title()} — {r['date']}")
            for t in r.get("topics", []):
                lines.append(f"  • {t}")
            if r.get("action_items"):
                lines.append("  Accionables:")
                for a in r["action_items"]:
                    lines.append(f"  → {a}")
        return "Últimos 1:1s:" + "\n".join(lines)
    except Exception as e:
        return f"Error al leer 1:1s: {str(e)}"

# ── NOTES ────────────────────────────────────────────────────────────────────

@tool
def create_note(title: str, content: str, category: str = "general") -> str:
    """Crea una nota general con título, contenido y categoría opcional."""
    def _fn():
        supabase.table("notes").insert({
            "title": title.strip(),
            "content": content.strip(),
            "category": category.lower().strip()
        }).execute()
    try:
        _retry(_fn)
        return f"✅ Nota guardada: {title}"
    except Exception as e:
        return f"Error al guardar nota: {str(e)}"

@tool
def get_notes(category: str = "", limit: int = 5) -> str:
    """Trae las últimas notas. Filtra por categoría si se especifica."""
    def _fn():
        query = supabase.table("notes") \
            .select("title, category, content, created_at") \
            .order("created_at", desc=True) \
            .limit(limit)
        if category:
            query = query.eq("category", category.lower().strip())
        return query.execute()
    try:
        result = _retry(_fn)
        if not result.data:
            return "No hay notas registradas."
        lines = []
        for r in result.data:
            lines.append(f"\n📝 {r['title']} [{r['category']}]")
            lines.append(f"   {r['content'][:200]}")
        return "Notas:" + "\n".join(lines)
    except Exception as e:
        return f"Error al leer notas: {str(e)}"

# ── TASKS ────────────────────────────────────────────────────────────────────

@tool
def create_task(title: str, description: str = "", priority: str = "medium", related_person: str = "", due_date: str = "") -> str:
    """Crea una tarea o pendiente. Priority: low, medium, high."""
    def _fn():
        data = {
            "title": title.strip(),
            "description": description.strip(),
            "priority": priority.lower(),
            "status": "pending",
            "related_person": related_person.lower().strip()
        }
        if due_date:
            data["due_date"] = due_date
        supabase.table("tasks").insert(data).execute()
    try:
        _retry(_fn)
        return f"✅ Tarea creada [{priority}]: {title}"
    except Exception as e:
        return f"Error al crear tarea: {str(e)}"

@tool
def get_tasks(status: str = "pending", limit: int = 10) -> str:
    """Trae las tareas. Status: pending, in_progress, done."""
    def _fn():
        query = supabase.table("tasks") \
            .select("title, description, status, priority, due_date, related_person") \
            .order("created_at", desc=True) \
            .limit(limit)
        if status:
            query = query.eq("status", status.lower())
        return query.execute()
    try:
        result = _retry(_fn)
        if not result.data:
            return f"No hay tareas con estado '{status}'."
        lines = []
        for r in result.data:
            due = f" — vence {r['due_date']}" if r.get("due_date") else ""
            person = f" ({r['related_person']})" if r.get("related_person") else ""
            lines.append(f"• [{r['priority'].upper()}] {r['title']}{person}{due}")
        return f"Tareas {status}:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error al leer tareas: {str(e)}"

@tool
def update_task_status(title: str, status: str) -> str:
    """Actualiza el estado de una tarea. Status: pending, in_progress, done."""
    def _fn():
        result = supabase.table("tasks") \
            .update({"status": status.lower(), "updated_at": "now()"}) \
            .ilike("title", f"%{title}%") \
            .execute()
        return result
    try:
        result = _retry(_fn)
        if result.data:
            return f"✅ Tarea '{title}' actualizada a '{status}'"
        return f"No encontré ninguna tarea que coincida con '{title}'"
    except Exception as e:
        return f"Error al actualizar tarea: {str(e)}"

# Lista de todos los tools para el agente
ALL_TOOLS = [
    create_one_on_one, get_one_on_ones,
    create_note, get_notes,
    create_task, get_tasks, update_task_status
]
