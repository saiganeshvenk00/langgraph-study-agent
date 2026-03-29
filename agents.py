from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field

from tools import read_note_file


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


class RouterOutput(BaseModel):
    subject: str = Field(description="One of: chemistry, physics, maths, history, geography")
    task: str = Field(description="One of: fetch_notes, summary, flashcards, rewrite")


SUBJECT_FILE_MAP = {
    "chemistry": "Chemistry.txt",
    "physics": "Physics.txt",
    "maths": "Maths.txt",
    "history": "History.txt",
    "geography": "Geography.txt"
}


def router_node(state):
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
        [SystemMessage(content=system_prompt)] + state["messages"]
    )

    return {
        "subject": result.subject.lower(),
        "task": result.task.lower(),
    }


def summary_agent(state):
    subject = state["subject"].lower()
    task = state["task"].lower()

    filename = SUBJECT_FILE_MAP.get(subject)

    if not filename:
        error_msg = f"Sorry, I couldn't find notes for the subject: {subject}"
        return {
            "output": error_msg,
            "messages": [AIMessage(content=error_msg)]
        }

    note_content = read_note_file(filename)

    if task == "fetch_notes":
        return {
            "filename": filename,
            "note_content": note_content,
            "output": note_content,
            "messages": [AIMessage(content=note_content)]
        }

    system = SystemMessage(content=f"""
You are a helpful study assistant.

Summarize the following notes into clear, concise study points.
Keep the summary easy to revise from.

Notes:
{note_content}
""")

    response = llm.invoke([system] + state["messages"])

    return {
        "filename": filename,
        "note_content": note_content,
        "output": response.content,
        "messages": [AIMessage(content=response.content)]
    }


def flashcard_agent(state):
    subject = state["subject"].lower()

    filename = SUBJECT_FILE_MAP.get(subject)

    if not filename:
        error_msg = f"Sorry, I couldn't find notes for the subject: {subject}"
        return {
            "output": error_msg,
            "messages": [AIMessage(content=error_msg)]
        }

    note_content = read_note_file(filename)

    system = SystemMessage(content=f"""
You are a helpful study assistant.

Convert the following notes into clear flashcards in Q/A format.
Keep them concise and useful for revision.
Generate 8 to 10 flashcards.

Notes:
{note_content}
""")

    response = llm.invoke([system] + state["messages"])

    return {
        "filename": filename,
        "note_content": note_content,
        "output": response.content,
        "messages": [AIMessage(content=response.content)]
    }


def rewrite_agent(state):
    subject = state["subject"].lower()

    filename = SUBJECT_FILE_MAP.get(subject)

    if not filename:
        error_msg = f"Sorry, I couldn't find notes for the subject: {subject}"
        return {
            "output": error_msg,
            "messages": [AIMessage(content=error_msg)]
        }

    note_content = read_note_file(filename)

    system = SystemMessage(content=f"""
You are a helpful study assistant.

Rewrite the following notes in simpler language for a beginner.
Make them easier to understand without losing the meaning.

Notes:
{note_content}
""")

    response = llm.invoke([system] + state["messages"])

    return {
        "filename": filename,
        "note_content": note_content,
        "output": response.content,
        "messages": [AIMessage(content=response.content)]
    }
