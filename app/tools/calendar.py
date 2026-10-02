import os
import json
import httpx
from datetime import datetime, timedelta
from langchain_core.tools import tool

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = "https://cto-hub.onrender.com/auth/google/callback"
SCOPES = "https://www.googleapis.com/auth/calendar.readonly"

# Token se guarda en memoria (en producción usarías Supabase)
_token_store: dict = {}

def get_auth_url() -> str:
    """Genera la URL de autorización de Google OAuth."""
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
    """Intercambia el código OAuth por tokens."""
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

async def refresh_token(refresh_tok: str) -> str:
    """Refresca el access token."""
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
    _token_store[chat_id] = tokens

def get_access_token(chat_id: str) -> str | None:
    tokens = _token_store.get(chat_id)
    if not tokens:
        return None
    return tokens.get("access_token")

def get_refresh_token(chat_id: str) -> str | None:
    tokens = _token_store.get(chat_id)
    if not tokens:
        return None
    return tokens.get("refresh_token")

async def get_calendar_events(chat_id: str, days: int = 7) -> list[dict]:
    """Trae los eventos del calendario de los próximos N días."""
    access_token = get_access_token(chat_id)
    if not access_token:
        return []

    now = datetime.utcnow()
    time_min = now.isoformat() + "Z"
    time_max = (now + timedelta(days=days)).isoformat() + "Z"

    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "timeMin": time_min,
                "timeMax": time_max,
                "singleEvents": True,
                "orderBy": "startTime",
                "maxResults": 20
            }
        )
        if r.status_code == 401:
            # Token expirado, refrescar
            refresh_tok = get_refresh_token(chat_id)
            if refresh_tok:
                new_token = await refresh_token(refresh_tok)
                _token_store[chat_id]["access_token"] = new_token
                r = await client.get(
                    "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                    headers={"Authorization": f"Bearer {new_token}"},
                    params={
                        "timeMin": time_min,
                        "timeMax": time_max,
                        "singleEvents": True,
                        "orderBy": "startTime",
                        "maxResults": 20
                    }
                )
        r.raise_for_status()
        return r.json().get("items", [])

def format_events(events: list[dict]) -> str:
    """Formatea eventos del calendario para Telegram."""
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
