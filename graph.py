from typing import Annotated, List, Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage

from agents import router_node, summary_agent, flashcard_agent, rewrite_agent, unknown_subject_node, unknown_task_node, list_notes_agent


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    subject: Annotated[str, "The subject of the study"]
    task: Annotated[str, "The task to be performed"]
    filename: Annotated[str, "The filename of the note"]
    note_content: Annotated[str, "The content of the note"]
    output: Annotated[str, "The output of the task"]
    session_id: Annotated[str, "The session id"]


def route_task(state: State):
    subject = state.get("subject", "").lower()
    task = state.get("task", "").lower()

    if subject == "unknown":
        return "unknown_subject_node"

    if task == "list_notes":
        return "list_notes_agent"

    if task in ["fetch_notes", "summary"]:
        return "summary_agent"
    elif task == "flashcards":
        return "flashcard_agent"
    elif task == "rewrite":
        return "rewrite_agent"

    return "unknown_task_node"


graph_builder = StateGraph[State, None, State, State](State)

graph_builder.add_node("router", router_node)
graph_builder.add_node("summary_agent", summary_agent)
graph_builder.add_node("flashcard_agent", flashcard_agent)
graph_builder.add_node("rewrite_agent", rewrite_agent)
graph_builder.add_node("unknown_subject_node", unknown_subject_node)
graph_builder.add_node("unknown_task_node", unknown_task_node)
graph_builder.add_node("list_notes_agent", list_notes_agent)

graph_builder.add_edge(START, "router")

graph_builder.add_conditional_edges(
    "router",
    route_task,
    {
        "summary_agent": "summary_agent",
        "flashcard_agent": "flashcard_agent",
        "rewrite_agent": "rewrite_agent",
        "unknown_subject_node": "unknown_subject_node",
        "unknown_task_node": "unknown_task_node",
        "list_notes_agent": "list_notes_agent",
    }
)

# the syntax for conditional edges is:
'''graph_builder.add_conditional_edges(
    source,        # arg 1
    path,          # arg 2
    path_map,      # arg 3 (optional)
)'''

graph_builder.add_edge("summary_agent", END)
graph_builder.add_edge("flashcard_agent", END)
graph_builder.add_edge("rewrite_agent", END)
graph_builder.add_edge("unknown_subject_node", END)
graph_builder.add_edge("unknown_task_node", END)
graph_builder.add_edge("list_notes_agent", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


def run_graph(user_input: str, session_id: str):
    state = graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "subject": "",
            "task": "",
            "filename": "",
            "note_content": "",
            "output": "",
            "session_id": session_id,
        },
        config={"configurable": {"thread_id": session_id}}
    )
    return state["output"]
