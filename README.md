# Study-Buddy 📚

An agentic AI study assistant built with **LangGraph**, **LangChain**, and **Gradio**. You point it at a folder of notes — in any format — and it can summarize them, create flashcards, rewrite them in simpler language, or just return the raw content. All powered by GPT-4.

---

## What It Does

You talk to it like a chatbot. It figures out what subject you're asking about and what you want done, then routes your request to the right agent automatically.

**Supported tasks:**
- `fetch notes` — returns the raw notes as-is
- `summarize` — condenses notes into clear study points
- `flashcards` — generates 8–10 Q/A flashcards from the notes
- `rewrite` — rewrites notes in simpler language for beginners
- `list notes` — tells you what subjects are available

**Supported file formats:**
- `.txt` — plain text
- `.pdf` — PDF documents
- `.docx` — Word documents
- `.pptx` — PowerPoint presentations

---

## How It Works — The Agentic Flow

This project is built around a **LangGraph state graph**. Every message you send flows through the graph like this:

```
User Message
     │
     ▼
┌─────────────┐
│   Router    │  ← LLM reads your message and extracts subject + task
└─────────────┘
     │
     ▼ (conditional routing based on subject + task)
     │
     ├──► summary_agent       (fetch_notes or summary)
     │
     ├──► flashcard_agent     (flashcards)
     │
     ├──► rewrite_agent       (rewrite)
     │
     ├──► list_notes_agent    (list_notes)
     │
     ├──► unknown_subject_node  (subject not found)
     │
     └──► unknown_task_node     (task not recognised)
           │
           ▼
         END  →  returns output to user
```

### Agents — What Each Node Does and Which Tools It Calls

| Agent / Node | Handles | Tool access |
|---|---|---|
| `router_node` | Every incoming message | Calls `list_note_files()` (via `get_subject_file_map()`) to build the dynamic subject list before the LLM decides routing |
| `summary_agent` | `fetch_notes` and `summary` tasks | Calls `list_note_files()` to resolve the subject → filename mapping, then `read_note_file()` to load the content. For `fetch_notes` the content is returned as-is; for `summary` it is passed to the LLM |
| `flashcard_agent` | `flashcards` task | Same file resolution pattern: `list_note_files()` + `read_note_file()`. Content is then passed to the LLM with a flashcard-generation prompt |
| `rewrite_agent` | `rewrite` task | Same file resolution pattern: `list_note_files()` + `read_note_file()`. Content is passed to the LLM with a simplification prompt |
| `list_notes_agent` | `list_notes` task | Calls `list_note_files()` only — no file content is read, just the filenames. Returns the subject list directly to the user |
| `unknown_subject_node` | Unrecognised subjects | Calls `list_note_files()` (via `get_subject_file_map()`) so it can tell the user which subjects *are* available |
| `unknown_task_node` | Unrecognised tasks | No tool calls — the LLM refuses and lists the supported tasks from its system prompt |

---

### Key Concept: State

Every node in the graph reads from and writes to a shared **State** object. This is how data flows between nodes without you having to pass variables around manually:

```python
class State(TypedDict):
    messages       # the conversation history
    subject        # extracted by router (e.g. "chemistry")
    task           # extracted by router (e.g. "flashcards")
    filename       # the file that was found for this subject
    note_content   # the raw text extracted from the file
    output         # the final response shown to the user
    session_id     # ties the conversation to a memory session
```

### Key Concept: Router

The router uses **structured output** — it forces the LLM to always return exactly a `subject` and a `task`, nothing else. This is what makes routing reliable:

```python
class RouterOutput(BaseModel):
    subject: str  # one of the allowed subjects or "unknown"
    task: str     # one of the allowed tasks or "unknown"
```

### Key Concept: Dynamic Subject Detection

There is no hardcoded list of subjects. The agent scans your Notes folder at runtime and builds the subject list from whatever files are there. Drop a new file in — it's automatically available:

```python
def get_subject_file_map() -> dict:
    files = list_note_files()
    return {f.rsplit(".", 1)[0].lower(): f for f in files}
```

---

## Project Structure

```
langgraph-study-agent/
│
├── main.py         # Gradio UI — folder picker, chat interface
├── graph.py        # LangGraph state graph — nodes, edges, routing
├── agents.py       # All agent functions + router node
├── tools.py        # File reading for txt, pdf, docx, pptx
├── requirements.txt
├── .env            # Your OpenAI API key goes here
│
└── Notes/          # Default folder for your note files
    ├── Chemistry.txt
    ├── Physics.txt
    ├── History.txt
    ├── Geography.txt
    └── Maths.txt
```

---

## Prerequisites — Accounts and Tools You Need

Before running this project, you need accounts and API keys from the following services:

### 1. OpenAI
This is the LLM powering all the agents.
- Sign up at [https://platform.openai.com](https://platform.openai.com)
- Go to **API Keys** and create a new secret key
- You need credits on your account for the API to work

### 2. LangSmith (optional but recommended)
LangSmith is a tracing and observability tool from LangChain. It lets you see exactly what your agents are doing — every LLM call, every prompt, every output — in a visual dashboard. Great for debugging.
- Sign up at [https://smith.langchain.com](https://smith.langchain.com)
- Go to **Settings → API Keys** and create a key
- Create a project (e.g. `Study-Buddy`) — this is where your traces will appear
- If you don't want tracing, just leave these keys out of your `.env` and set `LANGSMITH_TRACING=false`

### 3. Python 3.10+
Make sure you have Python installed. You can check with:
```bash
python --version
```
Download from [https://www.python.org](https://www.python.org) if needed.

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/saiganeshvenk00/langgraph-study-agent.git
cd langgraph-study-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your `.env` file

Create a file called `.env` in the root of the project and add your keys:

```
# Required — OpenAI API key
OPENAI_API_KEY=your-openai-key-here

# Optional — LangSmith tracing (set to false to disable)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your-langsmith-key-here
LANGSMITH_PROJECT="Study-Buddy"
```

> ⚠️ Never commit your `.env` file to git. It is already in `.gitignore` in this project.

### 5. Add your notes

Drop your notes into the `Notes/` folder. Any `.txt`, `.pdf`, `.docx`, or `.pptx` file works. The filename becomes the subject name — so `Biology.pdf` becomes the subject `biology`.

### 6. Run the app

```bash
python main.py
```

Open your browser at `http://127.0.0.1:7860`

---

## Using the App

1. **Set your Notes folder** — type the path or click `📂 Browse for Folder` and select it. Press Enter or click `Load Folder` to confirm.
2. **Chat** — type something like:
   - `"summarize my chemistry notes"`
   - `"give me flashcards for physics"`
   - `"rewrite history notes for a beginner"`
   - `"what subjects do I have?"`

---

## Example Conversation

```
You:  give me flashcards for linear algebra

Bot:  Q: What is a vector space?
      A: A set of vectors where addition and scalar multiplication are defined...

      Q: What is a basis?
      A: A linearly independent set that spans the vector space...
      ...
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | State graph framework for agentic flows |
| `langchain` | LLM chaining and message handling |
| `langchain-openai` | OpenAI LLM integration |
| `openai` | OpenAI API |
| `gradio` | Web UI |
| `pypdf` | Read PDF files |
| `python-docx` | Read Word documents |
| `python-pptx` | Read PowerPoint files |
| `python-dotenv` | Load `.env` variables |

---

## Tools (`tools.py`)

The agents never interact with the file system directly — they call these three tool functions defined in `tools.py`:

| Function | What it does |
|---|---|
| `list_note_files()` | Scans the Notes folder and returns a list of every supported file (`*.txt`, `*.pdf`, `*.docx`, `*.pptx`) |
| `read_note_file(file_name)` | Reads a single note file and returns its text content. Dispatches to the right parser based on file extension (`pypdf` for PDFs, `python-docx` for Word docs, `python-pptx` for PowerPoints, plain `open()` for text files) |
| `save_output(content, filename)` | Writes a string to a file on disk and returns the path. Used to persist generated output (summaries, flashcards, etc.) |

`set_notes_dir(path)` is also exposed so the Gradio UI can point the agent at a different folder at runtime without restarting.

---

## Things to Try Next

- Add more subjects by dropping files into the Notes folder
- Try asking for a summary of a PDF you uploaded
- Extend the router to support new task types (e.g. `quiz`, `mindmap`)
- Add a node that saves the output to a file automatically

---

## Built With

- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [LangChain](https://www.langchain.com/)
- [OpenAI GPT-4.1-mini](https://platform.openai.com/)
- [Gradio](https://www.gradio.app/)
