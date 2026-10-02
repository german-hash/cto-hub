import os
import httpx
import logging
from datetime import datetime
from app.tools.supabase_client import supabase, _retry
from app.tools.rag import ingest_text

logger = logging.getLogger(__name__)

NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
NOTION_API = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

# Páginas de Notion a importar
NOTION_PAGES = {
    "pablito":   "027d54e4877744e38a775a0ce06e8d4c",
    "tec_lead":  "285b7a3f36a98065bf3cd79aa55ebd89",
    "her":       "0759b73520e74ef8a099873182cb0b63",
    "nonides":   "d15383c7362745769125e0b881c9af85",
    "gallo":     "e942c31173d244b49ff0b2634665bc63",
    "carli":     "26b69c7edc83498f8e734bfa8ff0bfb4",
    "pablo_e":   "4236b83290814a92a77ab06b494c3bc5",
    "diego_m":   "06c580a5496d4532afc5b315d82c0e1a",
    "gonza":     "6e048b24cbcf4b12b1cad726f31e6eca",
    "caro_qa":   "987f4be1c02a49ad8f56355fe9deb4e0",
    "diego_c":   "360b7a3f36a98096a251fc40b34e43a3",
    "dce":       "bdaf76e99dc5438cb65bb951958a7e83",
    "tareas":    "34bb7a3f36a98017996de0cceeefb82f",
    "mis_notas": "35eb7a3f36a980618174edbdea501b5c",
}

def _get_rich_text(block: dict) -> str:
    btype = block.get("type", "")
    content = block.get(btype, {})
    rich_text = content.get("rich_text", [])
    return "".join([t.get("plain_text", "") for t in rich_text])

def _fetch_children(block_id: str, client: httpx.Client, depth: int = 0, max_depth: int = 4) -> list[str]:
    if depth > max_depth:
        return []
    lines = []
    try:
        r = client.get(
            f"{NOTION_API}/blocks/{block_id}/children",
            headers=HEADERS,
            params={"page_size": 100}
        )
        r.raise_for_status()
        blocks = r.json().get("results", [])
    except Exception as e:
        logger.error(f"Error fetching children {block_id}: {e}")
        return []

    for block in blocks:
        btype = block.get("type", "")
        text = _get_rich_text(block)
        has_children = block.get("has_children", False)

        if btype == "toggle":
            if text:
                lines.append(f"\n=== {text} ===")
            if has_children:
                lines.extend(_fetch_children(block["id"], client, depth + 1, max_depth))
        elif btype in ("bulleted_list_item", "numbered_list_item"):
            if text:
                lines.append(f"• {text}")
            if has_children:
                lines.extend(_fetch_children(block["id"], client, depth + 1, max_depth))
        elif btype.startswith("heading"):
            if text:
                lines.append(f"\n## {text}")
        elif btype == "paragraph":
            if text:
                lines.append(text)
            if has_children:
                lines.extend(_fetch_children(block["id"], client, depth + 1, max_depth))
        elif btype == "to_do":
            if text:
                checked = block.get("to_do", {}).get("checked", False)
                prefix = "✅" if checked else "⬜"
                lines.append(f"{prefix} {text}")

    return lines

def fetch_notion_page(page_id: str, name: str) -> str:
    """Lee el contenido completo de una página de Notion."""
    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(
                f"{NOTION_API}/blocks/{page_id}/children",
                headers=HEADERS,
                params={"page_size": 100}
            )
            r.raise_for_status()
            top_blocks = r.json().get("results", [])
            lines = _fetch_children.__wrapped__(page_id, client) if hasattr(_fetch_children, '__wrapped__') else []

            # Procesar bloques de primer nivel
            all_lines = []
            for block in top_blocks:
                btype = block.get("type", "")
                text = _get_rich_text(block)
                has_children = block.get("has_children", False)

                if btype == "toggle":
                    if text:
                        all_lines.append(f"\n=== {text} ===")
                    if has_children:
                        all_lines.extend(_fetch_children(block["id"], client))
                elif btype in ("bulleted_list_item", "numbered_list_item"):
                    if text:
                        all_lines.append(f"• {text}")
                    if has_children:
                        all_lines.extend(_fetch_children(block["id"], client))
                elif btype.startswith("heading"):
                    if text:
                        all_lines.append(f"\n## {text}")
                elif btype == "paragraph":
                    if text:
                        all_lines.append(text)
                elif btype == "to_do":
                    if text:
                        checked = block.get("to_do", {}).get("checked", False)
                        all_lines.append(f"{'✅' if checked else '⬜'} {text}")

            return "\n".join(all_lines)
    except Exception as e:
        logger.error(f"Error leyendo página {name}: {e}")
        return ""

def run_notion_import(pages: list[str] = None) -> dict:
    """
    Importa páginas de Notion a la knowledge base.
    Si pages es None, importa todas.
    Retorna un resumen de lo importado.
    """
    to_import = pages if pages else list(NOTION_PAGES.keys())
    results = {"success": [], "failed": [], "chunks_total": 0}

    for name in to_import:
        page_id = NOTION_PAGES.get(name)
        if not page_id:
            results["failed"].append(f"{name} (page_id no encontrado)")
            continue

        logger.info(f"Importando página: {name}")
        content = fetch_notion_page(page_id, name)

        if not content.strip():
            logger.warning(f"Página vacía: {name}")
            results["failed"].append(f"{name} (vacía)")
            continue

        # Ingestar en knowledge base con RAG
        chunks = ingest_text(
            content,
            source=f"notion/{name}",
            type="notion"
        )
        results["success"].append(name)
        results["chunks_total"] += chunks
        logger.info(f"✅ {name}: {chunks} chunks")

    return results
