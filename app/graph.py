from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from app.tools.memory import get_history, save_message, reset_history, get_memory
from app.agents.cto_agent import run_cto_agent

class HubState(TypedDict):
    chat_id: str
    user_message: str
    response: str

def load_context(state: HubState) -> HubState:
    """Carga historial y memoria antes de procesar."""
    return state

def cto_node(state: HubState) -> HubState:
    """Nodo principal: ejecuta el CTO Agent."""
    chat_id = state["chat_id"]
    user_message = state["user_message"]

    save_message(chat_id, "user", user_message)
    history = get_history(chat_id)
    memory = get_memory.invoke({})

    response = run_cto_agent(history, memory)
    save_message(chat_id, "assistant", response)

    return {**state, "response": response}

def build_graph():
    graph = StateGraph(HubState)
    graph.add_node("cto_agent", cto_node)
    graph.set_entry_point("cto_agent")
    graph.add_edge("cto_agent", END)
    return graph.compile()

hub = build_graph()

def process_message(chat_id: str, message: str) -> str:
    """Punto de entrada principal del hub."""
    result = hub.invoke({
        "chat_id": chat_id,
        "user_message": message,
        "response": ""
    })
    return result["response"]
