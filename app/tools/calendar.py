import os
import httpx
from datetime import datetime, timedelta
from app.tools.supabase_client import supabase, _retry

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = "https://cto-hub.onrender.com/auth/google/callback"
SCOPES = "https://www.googleapis.com/auth/calendar.readonly"

def get_auth_url() -> str:
    return (
        f"https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={SCOPES}"
        f"&access_type=offline"
        f"&prompt=consent"
    )

async def exchange_code(code: str) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post("https://oauth2.googleapis.com/token", data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code"
        })
        r.raise_for_status()
        return r.json()

async def _refresh_access_token(refresh_tok: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.post("https://oauth2.googleapis.com/token", data={
            "refresh_token": refresh_tok,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "grant_type": "refresh_token"
        })
        r.raise_for_status()
        return r.json()["access_token"]

def save_tokens(chat_id: str, tokens: dict):
    """Guarda tokens en Supabase."""
    def _fn():
        supabase.table("google_tokens").upsert({
            "chat_id": chat_id,
            "access_token": tokens.get("access_token", ""),
            "refresh_token": tokens.get("refresh_token", ""),
            "updated_at": "now()"
        }).execute()
    _retry(_fn)

def get_tokens(chat_id: str) -> dict | None:
    """Lee tokens desde Supabase."""
    def _fn():
        result = supabase.table("google_tokens") \
            .select("access_token, refresh_token") \
            .eq("chat_id", chat_id) \
            .execute()
        return result.data[0] if result.data else None
    try:
        return _retry(_fn)
    except Exception:
        return None

def get_access_token(chat_id: str) -> str | None:
    tokens = get_tokens(chat_id)
    return tokens.get("access_token") if tokens else None

async def get_calendar_events(chat_id: str, days: int = 7) -> list[dict]:
    tokens = get_tokens(chat_id)
    if not tokens:
        return []

    access_token = tokens["access_token"]
    refresh_tok = tokens.get("refresh_token", "")

    now = datetime.utcnow()
    time_min = now.isoformat() + "Z"
    time_max = (now + timedelta(days=days)).isoformat() + "Z"

    params = {
        "timeMin": time_min,
        "timeMax": time_max,
        "singleEvents": True,
        "orderBy": "startTime",
        "maxResults": 20
    }

    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            headers={"Authorization": f"Bearer {access_token}"},
            params=params
        )
        if r.status_code == 401 and refresh_tok:
            access_token = await _refresh_access_token(refresh_tok)
            save_tokens(chat_id, {"access_token": access_token, "refresh_token": refresh_tok})
            r = await client.get(
                "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                headers={"Authorization": f"Bearer {access_token}"},
                params=params
            )
        r.raise_for_status()
        return r.json().get("items", [])

def format_events(events: list[dict]) -> str:
    if not events:
        return "No tenés eventos en los próximos días."

    lines = []
    current_date = None

    for event in events:
        start = event.get("start", {})
        date_str = start.get("dateTime", start.get("date", ""))

        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            date_label = dt.strftime("%a %d/%m")
            time_label = dt.strftime("%H:%M")
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            date_label = dt.strftime("%a %d/%m")
            time_label = "Todo el día"

        if date_label != current_date:
            lines.append(f"\n📅 {date_label}")
            current_date = date_label

        title = event.get("summary", "Sin título")
        lines.append(f"  {time_label} — {title}")

    return "Agenda:" + "\n".join(lines)
