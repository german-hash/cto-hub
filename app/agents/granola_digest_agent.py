"""
Resumen diario de reuniones de Granola.

Trae las notas de Granola de un día (hora Argentina), arma un digest con Claude
y opcionalmente guarda los accionables en la tabla `tasks` de Supabase.
"""
import os
import re
import json
import time
import logging
from datetime import datetime, date, timedelta, timezone

import httpx
from anthropic import Anthropic

logger = logging.getLogger(__name__)

GRANOLA_BASE = "https://public-api.granola.ai/v1"
DIGEST_MODEL = os.environ.get("GRANOLA_DIGEST_MODEL", "claude-sonnet-5-5")
ART = timezone(timedelta(hours=-3))  # Argentina no tiene DST

MAX_NOTE_CHARS = 20000       # tope de texto por reunión que mandamos a Claude
MAX_TRANSCRIPT_CHARS = 3000  # tope de transcripción si no hay resumen


# ── Granola ──────────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ.get('GRANOLA_API_KEY', '')}"}


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _day_bounds_utc(day: date) -> tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=ART)
    end = start + timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def list_notes_for_day(day: date) -> list[dict]:
    """Lista las notas creadas en el día `day` (hora Argentina)."""
    start_utc, end_utc = _day_bounds_utc(day)
    params = {"created_after": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")}
    notes: list[dict] = []

    with httpx.Client(timeout=20) as client:
        for _ in range(10):  # máx 10 páginas, de sobra para un día
            r = client.get(f"{GRANOLA_BASE}/notes", headers=_headers(), params=params)
            r.raise_for_status()
            data = r.json()
            notes.extend(data.get("notes", []))
            cursor = data.get("cursor")
            if not data.get("hasMore") or not cursor:
                break
            params = {**params, "cursor": cursor}
            time.sleep(0.25)  # rate limit: 5 req/s

    result = []
    for n in notes:
        created = _parse_dt(n.get("created_at", ""))
        if created is None or start_utc <= created < end_utc:
            result.append(n)
    result.sort(key=lambda n: n.get("created_at", ""))
    return result


def _transcript_to_text(transcript) -> str:
    if not transcript:
        return ""
    if isinstance(transcript, dict):
        return transcript.get("text", "")
    if isinstance(transcript, list):
        lines = []
        for item in transcript:
            speaker = item.get("speaker") or {}
            label = speaker.get("diarization_label") or speaker.get("source") or ""
            text = item.get("text", "")
            lines.append(f"{label}: {text}" if label else text)
        return "\n".join(lines)
    return str(transcript)


def get_note_detail(note_id: str) -> dict | None:
    """Trae una nota completa. Si el transcript es muy grande, sigue sin él."""
    with httpx.Client(timeout=20) as client:
        r = client.get(f"{GRANOLA_BASE}/notes/{note_id}", headers=_headers(),
                       params={"include": "transcript"})
        if r.status_code == 413:
            r = client.get(f"{GRANOLA_BASE}/notes/{note_id}", headers=_headers())
        if r.status_code == 404:
            logger.info(f"Granola digest: nota {note_id} todavía no disponible")
            return None
        r.raise_for_status()
        return r.json()


def _note_to_text(note: dict) -> str:
    title = note.get("title") or "Reunión sin título"
    created = _parse_dt(note.get("created_at", ""))
    hour = created.astimezone(ART).strftime("%H:%M") if created else ""
    attendees = [a.get("name") or a.get("email", "") for a in note.get("attendees", []) or []]
    summary = (note.get("summary_markdown") or note.get("summary_text")
               or note.get("summary") or note.get("notes") or "")

    parts = [f"### {title}" + (f" ({hour})" if hour else "")]
    if attendees:
        parts.append("Participantes: " + ", ".join(a for a in attendees if a))
    if summary:
        parts.append("Resumen de Granola:\n" + summary)
    else:
        transcript = _transcript_to_text(note.get("transcript"))
        if transcript:
            parts.append("Transcripción (parcial):\n" + transcript[:MAX_TRANSCRIPT_CHARS])
    return "\n".join(parts)[:MAX_NOTE_CHARS]


# ── Claude ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Sos el asistente del CTO de Tecnología Digital de Arcos Dorados (McDonald's Argentina).
Te paso las notas de las reuniones de un día (de Granola). Armá un resumen ejecutivo en español rioplatense,
directo y sin relleno.

Formato de respuesta (respetalo exacto, sin texto antes ni después):
1) Primero el resumen para Telegram.
2) Después una línea que diga solo: ===ACCIONABLES===
3) Después un array JSON con los accionables:
[{"title": "...", "owner": "nombre o vacío", "due_date": "YYYY-MM-DD o vacío", "priority": "low|medium|high", "meeting": "título de la reunión"}]
Si no hay accionables, poné [].

Reglas para el resumen (texto plano, SIN markdown: nada de **, #, ni _; podés usar emojis y saltos de línea):
- Arrancá con "🗓️ Resumen de reuniones — DD/MM" y la cantidad de reuniones.
- Por cada reunión: "▪️ Título (hora)" y 1 a 3 bullets "•" con lo importante.
- Después una sección "✅ Decisiones" (solo si hubo).
- Después "📌 Accionables" con "→ tarea (responsable, fecha si hay)".
- Cerrá con "⚠️ Riesgos / temas abiertos" solo si hay algo que requiera atención del CTO.
- Máximo ~2500 caracteres en total.

Reglas para el JSON de accionables: solo compromisos concretos que surgen de las notas; no inventes responsables ni fechas.
"title" corto e imperativo, máximo 80 caracteres (el detalle ya está en el resumen).
Prioridad high solo si hay urgencia o impacto explícito."""

DELIMITER = "===ACCIONABLES==="


def _parse_action_items(raw: str) -> list[dict]:
    """Parsea el array JSON de accionables. Si viene cortado, rescata los objetos completos."""
    raw = raw.strip().strip("`").strip()
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    start = raw.find("[")
    if start == -1:
        return []
    raw = raw[start:]
    try:
        data = json.loads(raw[: raw.rfind("]") + 1])
        return [d for d in data if isinstance(d, dict)]
    except Exception:
        pass
    items = []
    for m in re.finditer(r"\{[^{}]*\}", raw):
        try:
            items.append(json.loads(m.group(0)))
        except Exception:
            continue
    return items


def build_digest(day: date, notes_text: list[str]) -> dict:
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    user_msg = f"Fecha: {day.strftime('%d/%m/%Y')}\nCantidad de reuniones: {len(notes_text)}\n\n" + "\n\n".join(notes_text)
    resp = client.messages.create(
        model=DIGEST_MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    if resp.stop_reason == "max_tokens":
        logger.warning("Granola digest: respuesta cortada por max_tokens")

    # Solo bloques de texto (ignora thinking)
    raw = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()

    if DELIMITER in raw:
        digest, actions_raw = raw.split(DELIMITER, 1)
        action_items = _parse_action_items(actions_raw)
    else:
        logger.warning("Granola digest: no vino el delimitador de accionables")
        digest, action_items = raw, []

    digest = digest.strip().strip("`").strip()
    return {"digest": digest or "⚠️ No pude armar el resumen.", "action_items": action_items}


# ── Supabase ─────────────────────────────────────────────────────────────────

def save_action_items(items: list[dict], day: date) -> int:
    """Guarda accionables en `tasks`. Evita duplicados si el job corre dos veces."""
    if not items:
        return 0
    from app.tools.supabase_client import supabase, _retry

    tag = f"[granola {day.isoformat()}]"

    def _existing():
        return supabase.table("tasks").select("title").ilike("description", f"{tag}%").execute()

    existing = {r["title"].strip().lower() for r in (_retry(_existing).data or [])}
    rows = []
    for it in items:
        title = (it.get("title") or "").strip()
        if not title or title.lower() in existing:
            continue
        row = {
            "title": title[:200],
            "description": f"{tag} {it.get('meeting', '')}".strip(),
            "priority": it.get("priority") if it.get("priority") in ("low", "medium", "high") else "medium",
            "status": "pending",
            "related_person": (it.get("owner") or "").lower().strip(),
        }
        if it.get("due_date"):
            row["due_date"] = it["due_date"]
        rows.append(row)

    if rows:
        _retry(lambda: supabase.table("tasks").insert(rows).execute())
    return len(rows)


# ── Entry point ──────────────────────────────────────────────────────────────

def run_granola_digest(day: date | None = None, create_tasks: bool = True) -> str:
    """Devuelve el texto del digest listo para mandar por Telegram."""
    if not os.environ.get("GRANOLA_API_KEY"):
        return "⚠️ GRANOLA_API_KEY no configurada."

    day = day or datetime.now(ART).date()
    try:
        notes = list_notes_for_day(day)
    except Exception as e:
        logger.error(f"Granola digest: error listando notas: {e}", exc_info=True)
        return f"⚠️ Error consultando Granola: {e}"

    if not notes:
        return f"🗓️ {day.strftime('%d/%m')}: no hay reuniones en Granola para resumir."

    notes_text = []
    for n in notes:
        try:
            detail = get_note_detail(n["id"])
            if detail:
                notes_text.append(_note_to_text(detail))
            time.sleep(0.25)
        except Exception as e:
            logger.warning(f"Granola digest: error trayendo {n.get('id')}: {e}")

    if not notes_text:
        return f"🗓️ {day.strftime('%d/%m')}: las reuniones todavía no tienen resumen en Granola."

    result = build_digest(day, notes_text)
    digest = result["digest"]

    if create_tasks and result["action_items"]:
        try:
            saved = save_action_items(result["action_items"], day)
            if saved:
                digest += f"\n\n💾 {saved} accionable(s) guardados en tareas."
        except Exception as e:
            logger.error(f"Granola digest: error guardando tareas: {e}", exc_info=True)

    return digest
