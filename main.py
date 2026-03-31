import os
import uuid
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


#this is the gradio UI - we use gr.Blocks so we can add the folder input on top of the chat interface
with gr.Blocks(title="Study-Buddy") as demo:
    gr.Markdown("# Study-Buddy")

    #the folder path input and the load button sit side by side in the same row
    with gr.Row():
        folder_input = gr.Textbox(
            label="Notes Folder Path",
            placeholder="e.g. C:/Users/you/Documents/Notes  or  Notes",
            value="Notes",
            scale=4
        )
        set_btn = gr.Button("Load Folder", scale=1)

    #this markdown box shows the confirmation message after the folder is loaded
    status = gr.Markdown("")

    #wire up the button click to the set_folder function
    set_btn.click(fn=set_folder, inputs=folder_input, outputs=status)

    #the chat interface sits below the folder section
    gr.ChatInterface(fn=chat)


if __name__ == "__main__":
    demo.launch()
