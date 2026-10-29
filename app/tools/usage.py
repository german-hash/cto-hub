"""
Registro de consumo de tokens por llamada a la API de Anthropic.

Guarda solo metadatos (fuente, modelo, tokens). Nunca guarda contenido de mensajes.
Nunca rompe el flujo: si falla el insert, solo loguea.
"""
import logging

logger = logging.getLogger(__name__)


def _get(obj, key, default=0):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default) or default
    return getattr(obj, key, default) or default


def log_usage(source: str, model: str, usage) -> None:
    """
    Acepta:
    - `response.usage` del SDK de Anthropic (input_tokens, output_tokens,
      cache_read_input_tokens, cache_creation_input_tokens)
    - `response.usage_metadata` de LangChain (input_tokens, output_tokens,
      input_token_details.cache_read / cache_creation)
    """
    try:
        details = _get(usage, "input_token_details", None)
        if details:  # formato LangChain: input_tokens ya incluye la caché
            cache_read = _get(details, "cache_read")
            # LangChain pone la escritura de caché en ephemeral_5m/1h y deja cache_creation en 0
            cache_creation = (_get(details, "cache_creation")
                              + _get(details, "ephemeral_5m_input_tokens")
                              + _get(details, "ephemeral_1h_input_tokens"))
            input_tokens = _get(usage, "input_tokens") - cache_read - cache_creation
        else:        # formato SDK de Anthropic: input_tokens excluye la caché
            cache_read = _get(usage, "cache_read_input_tokens")
            cache_creation = _get(usage, "cache_creation_input_tokens")
            input_tokens = _get(usage, "input_tokens")

        row = {
            "source": source,
            "model": model,
            "input_tokens": int(max(input_tokens, 0)),
            "output_tokens": int(_get(usage, "output_tokens")),
            "cache_read_tokens": int(cache_read),
            "cache_creation_tokens": int(cache_creation),
        }
        logger.info(f"usage {row}")

        from app.tools.supabase_client import supabase
        supabase.table("api_usage").insert(row).execute()
    except Exception as e:
        logger.warning(f"No pude registrar uso de tokens: {e}")
