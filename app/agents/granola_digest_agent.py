"""
Resumen SEMANAL de reuniones de Granola.

Trae las notas de Granola de una semana (lunes a domingo, hora Argentina),
arma un resumen de grandes temas + pendientes para la semana siguiente con Claude
y opcionalmente guarda los pendientes en la tabla `tasks` de Supabase.
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

MAX_NOTE_CHARS = 8000        # tope por reunión (en una semana pueden ser 20-40 reuniones)
MAX_TRANSCRIPT_CHARS = 3000  # tope de transcripción si no hay resumen
DELIMITER = "===PENDIENTES==="


# ── Fechas ───────────────────────────────────────────────────────────────────

def week_bounds(any_day: date) -> tuple[date, date]:
    """Lunes y domingo de la semana que contiene `any_day`."""
    monday = any_day - timedelta(days=any_day.weekday())
    return monday, monday + timedelta(days=6)


def _range_utc(start_day: date, end_day: date) -> tuple[datetime, datetime]:
    start = datetime(start_day.year, start_day.month, start_day.day, tzinfo=ART)
    end = datetime(end_day.year, end_day.month, end_day.day, tzinfo=ART) + timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


# ── Granola ──────────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ.get('GRANOLA_API_KEY', '')}"}


def list_notes_for_range(start_day: date, end_day: date) -> list[dict]:
    """Lista las notas creadas entre start_day y end_day inclusive (hora Argentina)."""
    start_utc, end_utc = _range_utc(start_day, end_day)
    params = {"created_after": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")}
    notes: list[dict] = []

    with httpx.Client(timeout=20) as client:
        for _ in range(20):  # máx 20 páginas
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


def get_note_detail(note_id: str, with_transcript: bool = False) -> dict | None:
    params = {"include": "transcript"} if with_transcript else {}
    with httpx.Client(timeout=20) as client:
        r = client.get(f"{GRANOLA_BASE}/notes/{note_id}", headers=_headers(), params=params)
        if r.status_code == 413:
            r = client.get(f"{GRANOLA_BASE}/notes/{note_id}", headers=_headers())
        if r.status_code == 404:
            logger.info(f"Granola weekly: nota {note_id} todavía no disponible")
            return None
        r.raise_for_status()
        return r.json()


def _summary_of(note: dict) -> str:
    return (note.get("summary_markdown") or note.get("summary_text")
            or note.get("summary") or note.get("notes") or "")


def _note_to_text(note: dict) -> str:
    title = note.get("title") or "Reunión sin título"
    created = _parse_dt(note.get("created_at", ""))
    when = created.astimezone(ART).strftime("%d/%m %H:%M") if created else ""
    attendees = [a.get("name") or a.get("email", "") for a in note.get("attendees", []) or []]

    parts = [f"### {title}" + (f" — {when}" if when else "")]
    if attendees:
        parts.append("Participantes: " + ", ".join(a for a in attendees if a))
    summary = _summary_of(note)
    if summary:
        parts.append(summary)
    else:
        transcript = _transcript_to_text(note.get("transcript"))
        if transcript:
            parts.append("Transcripción (parcial):\n" + transcript[:MAX_TRANSCRIPT_CHARS])
    return "\n".join(parts)[:MAX_NOTE_CHARS]


# ── Claude ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Sos el asistente del CTO de Tecnología Digital de Arcos Dorados (McDonald's Argentina).
Te paso las notas de TODAS las reuniones de una semana (de Granola). Quiero una visión de alto nivel,
no un resumen reunión por reunión. Español rioplatense, directo y sin relleno.

Formato de respuesta (respetalo exacto, sin texto antes ni después):
1) Primero el resumen para Telegram.
2) Después una línea que diga solo: ===PENDIENTES===
3) Después un array JSON con los pendientes para la semana próxima:
[{"title": "...", "owner": "nombre o vacío", "due_date": "YYYY-MM-DD o vacío", "priority": "low|medium|high", "theme": "tema al que pertenece"}]
Si no hay pendientes, poné [].

Reglas para el resumen (texto plano, SIN markdown: nada de **, #, ni _; podés usar emojis y saltos de línea):
- Arrancá con "📅 Resumen semanal — DD/MM al DD/MM" y en la línea siguiente "N reuniones".
- "🎯 Grandes temas": entre 3 y 6 temas, agrupando lo que se habló en distintas reuniones.
  Cada tema: "▪️ Nombre del tema" y 1 o 2 líneas con dónde está parado y qué se decidió.
  Nada de detalle operativo menor; priorizá lo que importa a nivel CTO.
- "📌 Pendientes para la semana que viene": máximo 8, los más importantes,
  "→ pendiente (responsable)". Agregá fecha solo si es un compromiso explícito.
- "⚠️ Ojo con": máximo 3 riesgos o bloqueos que requieran atención del CTO. Omití la sección si no hay.
- Máximo ~1800 caracteres en total.

Reglas para el JSON de pendientes: los mismos de la sección de pendientes (máximo 8), sin inventar responsables ni fechas.
"title" corto e imperativo, máximo 80 caracteres. Prioridad high solo si hay urgencia o impacto explícito."""


def _parse_items(raw: str) -> list[dict]:
    """Parsea el array JSON. Si viene cortado, rescata los objetos completos."""
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


def build_weekly_digest(start_day: date, end_day: date, notes_text: list[str]) -> dict:
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    user_msg = (f"Semana: {start_day.strftime('%d/%m/%Y')} al {end_day.strftime('%d/%m/%Y')}\n"
                f"Cantidad de reuniones: {len(notes_text)}\n\n" + "\n\n".join(notes_text))
    resp = client.messages.create(
        model=DIGEST_MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    from app.tools.usage import log_usage
    log_usage("granola_weekly", DIGEST_MODEL, resp.usage)
    if resp.stop_reason == "max_tokens":
        logger.warning("Granola weekly: respuesta cortada por max_tokens")

    raw = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()

    if DELIMITER in raw:
        digest, items_raw = raw.split(DELIMITER, 1)
        items = _parse_items(items_raw)
    else:
        logger.warning("Granola weekly: no vino el delimitador de pendientes")
        digest, items = raw, []

    digest = digest.strip().strip("`").strip()
    return {"digest": digest or "⚠️ No pude armar el resumen.", "action_items": items}


# ── Supabase ─────────────────────────────────────────────────────────────────

def save_action_items(items: list[dict], week_start: date) -> int:
    """Guarda pendientes en `tasks`. Evita duplicados si el job corre dos veces la misma semana."""
    if not items:
        return 0
    from app.tools.supabase_client import supabase, _retry

    tag = f"[granola semana {week_start.isoformat()}]"

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
            "description": f"{tag} {it.get('theme', '')}".strip(),
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

def run_granola_weekly_digest(any_day: date | None = None, create_tasks: bool = True) -> str:
    """Resumen de la semana (lun-dom) que contiene `any_day`. Por defecto, la semana actual."""
    if not os.environ.get("GRANOLA_API_KEY"):
        return "⚠️ GRANOLA_API_KEY no configurada."

    today = datetime.now(ART).date()
    start_day, end_day = week_bounds(any_day or today)
    if start_day <= today <= end_day:
        end_day = today  # semana en curso: hasta hoy
    label = f"{start_day.strftime('%d/%m')} al {end_day.strftime('%d/%m')}"

    try:
        notes = list_notes_for_range(start_day, end_day)
    except Exception as e:
        logger.error(f"Granola weekly: error listando notas: {e}", exc_info=True)
        return f"⚠️ Error consultando Granola: {e}"

    if not notes:
        return f"📅 Semana {label}: no hay reuniones en Granola para resumir."

    notes_text = []
    for n in notes:
        try:
            detail = get_note_detail(n["id"])
            if detail and not _summary_of(detail):
                detail = get_note_detail(n["id"], with_transcript=True) or detail
            if detail:
                notes_text.append(_note_to_text(detail))
            time.sleep(0.25)
        except Exception as e:
            logger.warning(f"Granola weekly: error trayendo {n.get('id')}: {e}")

    if not notes_text:
        return f"📅 Semana {label}: las reuniones todavía no tienen resumen en Granola."

    logger.info(f"Granola weekly: {len(notes_text)} reuniones, {sum(map(len, notes_text))} chars")
    result = build_weekly_digest(start_day, end_day, notes_text)
    digest = result["digest"]

    if create_tasks and result["action_items"]:
        try:
            saved = save_action_items(result["action_items"], start_day)
            if saved:
                digest += f"\n\n💾 {saved} pendiente(s) guardados en tareas."
        except Exception as e:
            logger.error(f"Granola weekly: error guardando tareas: {e}", exc_info=True)

    return digest
