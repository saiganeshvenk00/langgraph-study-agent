from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field

from tools import read_note_file, list_note_files


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


class RouterOutput(BaseModel):
    subject: str = Field(description="One of: chemistry, physics, maths, history, geography, unknown")
    task: str = Field(description="One of: fetch_notes, summary, flashcards, rewrite, list_notes, unknown")


def get_subject_file_map() -> dict:
    """Dynamically build subject->filename map from whatever files exist in Notes/."""
    files = list_note_files()
    return {f.rsplit(".", 1)[0].lower(): f for f in files}


def router_node(state):
    router_llm = llm.with_structured_output(RouterOutput)

    available_subjects = ", ".join(get_subject_file_map().keys())

    system_prompt = f"""
You are a strict router for a StudyHelp chatbot.

Your job is to extract:
1. subject
2. task

Allowed subjects (determined by files currently in the Notes folder):
{available_subjects}
- unknown  (use this if the subject is not in the list above, or is unclear)

Allowed tasks:
- fetch_notes  (user asks for notes, study material, chapter, or content)
- summary      (user asks to summarize)
- flashcards   (user asks for flashcards, Q/A, or quiz cards)
- rewrite      (user asks to rewrite, simplify, or explain in easier words)
- list_notes   (user asks what subjects/notes are available, what is in the folder, or what they can study)
- unknown      (use this if the task does not match any of the above)

Rules:
- Never guess a subject. If it is not clearly one of the allowed subjects listed above, return "unknown".
- Never guess a task. If it does not clearly match an allowed task, return "unknown".
- Do not map unrecognised subjects to the closest subject.
- Return only the exact allowed values listed above.
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

    filename = get_subject_file_map().get(subject)

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

    filename = get_subject_file_map().get(subject)

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

    filename = get_subject_file_map().get(subject)

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


def unknown_subject_node(state):
    available = ", ".join(get_subject_file_map().keys())
    system = SystemMessage(content=f"""
You are a helpful study assistant.

The user asked about a subject that does not exist in the notes library.
Politely let them know their subject is not available and tell them the available subjects are: {available}.
Do not make up or suggest notes for unavailable subjects.
""")
    response = llm.invoke([system] + state["messages"])
    return {
        "output": response.content,
        "messages": [AIMessage(content=response.content)]
    }


def unknown_task_node(state):
    system = SystemMessage(content="""
You are a strict study assistant. You MUST NOT perform any task other than:
fetch_notes, summary, flashcards, rewrite, or list_notes.
The user has requested something outside these tasks.
You MUST refuse and tell them what you can help with. Do not fulfil their request under any circumstances.
""")
    response = llm.invoke([system] + state["messages"])
    return {
        "output": response.content,
        "messages": [AIMessage(content=response.content)]
    }

def list_notes_agent(state):
    files = list_note_files()
    subjects = [f.rsplit(".", 1)[0] for f in files]
    msg = f"Here are the available subjects: {', '.join(subjects)}. I can fetch notes, summarize, create flashcards, or rewrite them for any of these."
    return {"output": msg, "messages": [AIMessage(content=msg)]}