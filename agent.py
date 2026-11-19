from typing import Annotated, Literal
from typing_extensions import TypedDict
import requests
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# გლობალური სია დემონსტრაციისთვის (CLI-სთვის); Streamlit იყენებს session_state-საც
tasks_db = []

@tool
def add_task(task: str) -> str:
    """Adds a new study task to the task list."""
    tasks_db.append(task)
    return f"Task added: '{task}'"

@tool
def list_tasks() -> str:
    """Returns all current tasks in the task list."""
    if not tasks_db:
        return "Your task list is currently empty."
    return "Your tasks:\n" + "\n".join(f"- {t}" for t in tasks_db)

@tool
def delete_task(task: str) -> str:
    """Deletes a specific task from the task list."""
    if task in tasks_db:
        tasks_db.remove(task)
        return f"Task deleted: '{task}'"
    return f"Task '{task}' not found in the list."

@tool
def get_tbilisi_weather() -> dict:
    """Get current temperature in Tbilisi, in Celsius."""
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": 41.72, "longitude": 44.79, "current": "temperature_2m"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()["current"]

@tool
def search_books(topic: str) -> list:
    """Search for study books on a given topic using Open Library."""
    response = requests.get(
        "https://openlibrary.org/search.json",
        params={"q": topic, "limit": 3},
        timeout=20,
    )
    response.raise_for_status()
    docs = response.json().get("docs", [])
    results = []
    for doc in docs:
        title = doc.get("title", "Unknown title")
        authors = ", ".join(doc.get("author_name", ["Unknown author"]))
        results.append(f"{title} — {authors}")
    return results

class State(TypedDict):
    messages: Annotated[list, add_messages]

def build_graph():
    safe_tools = [add_task, list_tasks, get_tbilisi_weather, search_books]
    sensitive_tools = [delete_task]
    all_tools = safe_tools + sensitive_tools

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    llm_with_tools = llm.bind_tools(all_tools)

    def assistant(state: State):
        system_msg = (
            "You are Study Day Assistant. Help the user manage study tasks, "
            "check Tbilisi weather, and search for books. Be concise and helpful."
        )
        messages = [{"role": "system", "content": system_msg}] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def route_tools(state: State) -> Literal["safe_tools", "sensitive_tools", "__end__"]:
        last_message = state["messages"][-1]
        if not getattr(last_message, "tool_calls", None):
            return "__end__"
        for tc in last_message.tool_calls:
            if tc["name"] == "delete_task":
                return "sensitive_tools"
        return "safe_tools"

    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_node("safe_tools", ToolNode(safe_tools))
    builder.add_node("sensitive_tools", ToolNode(sensitive_tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", route_tools)
    builder.add_edge("safe_tools", "assistant")
    builder.add_edge("sensitive_tools", "assistant")

    memory = MemorySaver()
    return builder.compile(checkpointer=memory, interrupt_before=["sensitive_tools"])
