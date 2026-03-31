import os
import uuid
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
from graph import run_graph
import tools


#generate a unique session id when the app starts so the conversation memory is tied to this session
session_id = str(uuid.uuid4())


#this definition is the main chat function that gradio calls whenever the user sends a message
def chat(message, history):
    return run_graph(message, session_id)


#this definition is called when the user clicks the Load Folder button - it updates the notes directory in tools.py
def set_folder(path):
    if not path.strip():
        return "⚠️ Please enter a valid folder path."
    tools.set_notes_dir(path.strip())
    return f"✅ **Loaded:** `{path.strip()}`"


#this definition is called when the user selects something in the file explorer
#if they picked a file, we use its parent folder - if they picked a folder, we use it directly
def on_explorer_select(selected):
    if not selected:
        return gr.update(), ""
    path = selected[0] if isinstance(selected, list) else selected
    p = Path(path)
    folder = str(p) if p.is_dir() else str(p.parent)
    msg = set_folder(folder)
    return folder, msg


#this is the gradio UI - we use gr.Blocks so we can add the folder input on top of the chat interface
with gr.Blocks(title="Study-Buddy") as demo:
    gr.Markdown("# Study-Buddy")

    #the folder path input and load button sit together so the user knows they are related
    with gr.Row(equal_height=True):
        folder_input = gr.Textbox(
            label="Notes Folder Path (Type a path and press Enter, or Browse for Folder below)",
            placeholder="Enter folder path here...",
            value="",
            scale=4
        )
        set_btn = gr.Button("Load Folder", scale=1, min_width=120)

    #the browse button sits on its own row below, clearly separate from the load action
    with gr.Row():
        browse_btn = gr.Button("📂 Browse for Folder", scale=1)

    #this markdown box shows the confirmation message after the folder is loaded
    status = gr.Markdown("")

    #the file explorer is hidden by default and only shows when the user clicks Browse
    explorer = gr.FileExplorer(
        label="Select your Notes folder (selecting any file inside it works too)",
        root_dir=str(Path.home()),
        file_count="single",
        interactive=True,
        visible=False,
        height=200,
    )

    #wire up the browse button to toggle the explorer visibility
    browse_btn.click(fn=lambda: gr.update(visible=True), inputs=None, outputs=explorer)

    #wire up the button click to the set_folder function
    set_btn.click(fn=set_folder, inputs=folder_input, outputs=status)

    #wire up the Enter key on the textbox to do the same thing as clicking Load Folder
    folder_input.submit(fn=set_folder, inputs=folder_input, outputs=status)

    #wire up the explorer - when the user selects something, auto-fill the text box, load the folder, and hide the explorer again
    explorer.change(fn=on_explorer_select, inputs=explorer, outputs=[folder_input, status])

    #the chat interface sits below the folder section
    gr.ChatInterface(fn=chat)


if __name__ == "__main__":
    demo.launch()
