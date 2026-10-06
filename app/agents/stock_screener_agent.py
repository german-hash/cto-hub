import os
from datetime import date
from anthropic import Anthropic
import yfinance as yf

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

ACCIONES_EEUU = [
    "AAPL", "MSFT", "GOOGL", "META", "BRK-B", "JNJ", "JPM", "XOM",
    "PG", "V", "UNH", "HD", "CVX", "MRK", "ABBV", "PEP", "KO", "BAC"
]
ACCIONES_EUROPA = [
    "ASML.AS", "NESN.SW", "NOVN.SW", "ROG.SW", "SAP.DE", "SIE.DE",
    "TTE.PA", "LVMH.PA", "OR.PA", "AZN.L", "HSBA.L", "BP.L"
]
ACCIONES_CHINA = [
    "BABA", "TCEHY", "JD", "BIDU", "NIO", "PDD", "NTES", "VIPS"
]

def obtener_datos_accion(symbol: str) -> dict | None:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        cashflow = ticker.cashflow
        fcf_actual = fcf_anterior = 0
        if cashflow is not None and not cashflow.empty:
            if "Free Cash Flow" in cashflow.index:
                fcf_values = cashflow.loc["Free Cash Flow"].dropna()
                if len(fcf_values) >= 1:
                    fcf_actual = float(fcf_values.iloc[0])
                if len(fcf_values) >= 2:
                    fcf_anterior = float(fcf_values.iloc[1])
        return {
            "symbol": symbol,
            "nombre": info.get("longName", symbol),
            "precio": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            "pe": info.get("trailingPE"),
            "market_cap": info.get("marketCap", 0),
            "fcf_actual": fcf_actual,
            "fcf_anterior": fcf_anterior,
            "pais": info.get("country", "N/A")
        }
    except Exception:
        return None

def run_stock_screener_agent() -> str:
    """Ejecuta el screener de acciones y devuelve el TOP 10."""
    hoy = date.today().strftime("%d/%m/%Y")
    todas = ACCIONES_EEUU + ACCIONES_EUROPA + ACCIONES_CHINA

    candidatas = []
    for symbol in todas:
        datos = obtener_datos_accion(symbol)
        if not datos:
            continue
        market_cap = datos["market_cap"] or 0
        pe = datos["pe"]
        fcf_actual = datos["fcf_actual"]
        fcf_anterior = datos["fcf_anterior"]

        try:
            pe_valor = float(pe) if pe is not None else None
        except (ValueError, TypeError):
            pe_valor = None

        if (market_cap > 10_000_000_000 and
                pe_valor is not None and 0 < pe_valor < 25 and
                fcf_actual > 0 and fcf_actual >= fcf_anterior):
            candidatas.append(datos)

    if not candidatas:
        return "No se encontraron acciones que cumplan todos los criterios hoy."

    resumen = f"Fecha: {hoy}\n\nAcciones Large Cap + PE < 25 + FCF positivo creciente:\n\n"
    for a in candidatas:
        resumen += f"- {a['symbol']} ({a['nombre']}) [{a['pais']}]: "
        resumen += f"Precio: ${a['precio']:.2f}, "
        try:
            resumen += f"PE: {float(a['pe']):.1f}, " if a['pe'] else "PE: N/A, "
        except Exception:
            resumen += "PE: N/A, "
        resumen += f"Market Cap: ${a['market_cap']/1e9:.1f}B, "
        resumen += f"FCF actual: ${a['fcf_actual']/1e9:.2f}B, "
        resumen += f"FCF anterior: ${a['fcf_anterior']/1e9:.2f}B\n"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system="""Eres un analista financiero experto.
Con los datos proporcionados, seleccionas las TOP 10 acciones del dia.
Criterios: Large Cap, PE bajo, Free Cash Flow positivo y creciente.
Para cada accion presentas: nombre, simbolo, precio, PE, Market Cap y una linea explicando por que la elegiste.
El formato debe ser claro y conciso, ideal para leer en Telegram.
Respondés siempre en español.""",
        messages=[{"role": "user", "content": f"Analizá estas acciones y seleccioná el TOP 10:\n\n{resumen}"}]
    )
    return response.content[0].text
