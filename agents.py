from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field

from tools import read_note_file, list_note_files


#load the environment variables from the .env file so we can access the OpenAI API key
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

#this is the LLM we are using for all agents - temperature 0 means it gives consistent, non-random answers
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


#this is the structured output model for the router - it forces the LLM to always return a subject and a task
class RouterOutput(BaseModel):
    subject: str = Field(description="One of: chemistry, physics, maths, history, geography, unknown")
    task: str = Field(description="One of: fetch_notes, summary, flashcards, rewrite, list_notes, unknown")


#this definition is used to dynamically build the subject to filename map by scanning whatever files are currently in the Notes folder
#this means you dont have to hardcode the subjects - just drop a file in and it gets picked up automatically
def get_subject_file_map() -> dict:
    files = list_note_files()
    return {f.rsplit(".", 1)[0].lower(): f for f in files}


#this is the router node - it reads the users message and figures out what subject they are asking about and what task they want done
#it uses structured output so the LLM is forced to return only allowed values and not make things up
def router_node(state):
    router_llm = llm.with_structured_output(RouterOutput)

    #build the allowed subjects list dynamically from whatever files are in the notes folder right now
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


#this agent handles both fetch_notes and summary tasks
#if the task is fetch_notes it just returns the raw notes, if its summary it asks the LLM to summarize them
def summary_agent(state):
    subject = state["subject"].lower()
    task = state["task"].lower()

    #look up the filename for this subject from the notes folder
    filename = get_subject_file_map().get(subject)

    #if no file was found for this subject, return an error message
    if not filename:
        error_msg = f"Sorry, I couldn't find notes for the subject: {subject}"
        return {
            "output": error_msg,
            "messages": [AIMessage(content=error_msg)]
        }

    #read the content of the file regardless of whether its txt, pdf, docx or pptx
    note_content = read_note_file(filename)

    #if the user just wants the raw notes, return them as is without calling the LLM
    if task == "fetch_notes":
        return {
            "filename": filename,
            "note_content": note_content,
            "output": note_content,
            "messages": [AIMessage(content=note_content)]
        }

    #if the task is summary, pass the notes to the LLM and ask it to summarize
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


#this agent handles the flashcards task - it takes the notes and turns them into Q/A style flashcards
def flashcard_agent(state):
    subject = state["subject"].lower()

    #look up the filename for this subject from the notes folder
    filename = get_subject_file_map().get(subject)

    #if no file was found for this subject, return an error message
    if not filename:
        error_msg = f"Sorry, I couldn't find notes for the subject: {subject}"
        return {
            "output": error_msg,
            "messages": [AIMessage(content=error_msg)]
        }

    #read the content of the file regardless of whether its txt, pdf, docx or pptx
    note_content = read_note_file(filename)

    #pass the notes to the LLM and ask it to generate flashcards
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


#this agent handles the rewrite task - it takes the notes and rewrites them in simpler language for beginners
def rewrite_agent(state):
    subject = state["subject"].lower()

    #look up the filename for this subject from the notes folder
    filename = get_subject_file_map().get(subject)

    #if no file was found for this subject, return an error message
    if not filename:
        error_msg = f"Sorry, I couldn't find notes for the subject: {subject}"
        return {
            "output": error_msg,
            "messages": [AIMessage(content=error_msg)]
        }

    #read the content of the file regardless of whether its txt, pdf, docx or pptx
    note_content = read_note_file(filename)

    #pass the notes to the LLM and ask it to rewrite them in simpler language
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


#this node handles the case where the user asks about a subject that doesnt exist in the notes folder
#it tells the user what subjects are actually available
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


#this node handles the case where the user asks for something that is not one of the allowed tasks
#it refuses to help and tells the user what it can actually do
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


#this agent handles the list_notes task - it just lists all the subjects available in the notes folder
def list_notes_agent(state):
    files = list_note_files()
    subjects = [f.rsplit(".", 1)[0] for f in files]
    msg = f"Here are the available subjects: {', '.join(subjects)}. I can fetch notes, summarize, create flashcards, or rewrite them for any of these."
    return {"output": msg, "messages": [AIMessage(content=msg)]}
