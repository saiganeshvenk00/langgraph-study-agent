import os
import uuid
import gradio as gr

from typing import Annotated, List, Any, Optional, Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tools import list_note_files, read_note_file, save_output

from pydantic import BaseModel, Field


load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

#Agent Class Definitions
class RouterOutput(BaseModel):
    subject: str = Field(description="One of: chemistry, physics, maths, history, geography")
    task: str = Field(description="One of: fetch_notes, summary, flashcards, rewrite")




#this is a map of the subject and the file name
SUBJECT_FILE_MAP = {
    "chemistry": "Chemistry.txt",
    "physics": "Physics.txt",
    "maths": "Maths.txt",
    "history": "History.txt",
    "geography": "Geography.txt"
}

#Step1: Define the state of the graph

class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    subject: Annotated[str, "The subject of the study"]
    task: Annotated[str, "The task to be performed"]
    filename: Annotated[str, "The filename of the note"]
    note_content: Annotated[str, "The content of the note"]
    output: Annotated[str, "The output of the task"]
    session_id: Annotated[str, "The session id"]

#Step2: Define the graph
graph_builder= StateGraph[State, None, State, State](State)

#Step3: Define the nodes------------------------------------------------------

#Router Node
def router_node(state: State):
    latest_message = state["messages"][-1]

    router_llm = llm.with_structured_output(RouterOutput)

    system_prompt = """
You are a strict router for a StudyHelp chatbot.

Your job is to extract:
1. subject
2. task

Allowed subjects:
- chemistry
- physics
- maths
- history
- geography

Allowed tasks:
- fetch_notes
- summary
- flashcards
- rewrite

Rules:
- If the user asks for notes, study material, chapter, or content directly, use fetch_notes.
- If the user asks to summarize, use summary.
- If the user asks for flashcards, Q/A, or quiz cards, use flashcards.
- If the user asks to rewrite, simplify, or explain in easier words, use rewrite.
- Return only allowed values.
- If the subject is unclear, choose the closest valid subject only if it is obvious, or ask the user to clarify.
- If the task is unclear, choose the closest valid task only if it is obvious, or ask the user to clarify.
"""

    result = router_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=latest_message.content),
        ]
    )

    return {
        "subject": result.subject.lower(),
        "task": result.task.lower(),
    }

#Summary Node
def summary_agent(state: State):
    subject = state["subject"].lower()
    task = state["task"].lower()

    filename = SUBJECT_FILE_MAP.get(subject)

    if not filename:
        return {
            "output": f"Sorry, I couldn't find notes for the subject: {subject}"
        }

    note_content = read_note_file(filename)

    if task == "fetch_notes":
        return {
            "filename": filename,
            "note_content": note_content,
            "output": note_content
        }

    prompt = f"""
You are a helpful study assistant.

Summarize the following notes into clear, concise study points.
Keep the summary easy to revise from.

Notes:
{note_content}
"""

    response = llm.invoke(prompt)

    return {
        "filename": filename,
        "note_content": note_content,
        "output": response.content
    }

