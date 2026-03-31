import os
import uuid
import gradio as gr
from dotenv import load_dotenv
from graph import run_graph
import tools


session_id = str(uuid.uuid4())


def chat(message, history):
    return run_graph(message, session_id)


def set_folder(path):
    if not path.strip():
        return "⚠️ Please enter a valid folder path."
    tools.set_notes_dir(path.strip())
    return f"✅ **Loaded:** `{path.strip()}`"


with gr.Blocks(title="Study-Buddy") as demo:
    gr.Markdown("# Study-Buddy")

    with gr.Row():
        folder_input = gr.Textbox(
            label="Notes Folder Path",
            placeholder="e.g. C:/Users/you/Documents/Notes  or  Notes",
            value="Notes",
            scale=4
        )
        set_btn = gr.Button("Load Folder", scale=1)

    status = gr.Markdown("")

    set_btn.click(fn=set_folder, inputs=folder_input, outputs=status)

    gr.ChatInterface(fn=chat)


if __name__ == "__main__":
    demo.launch()
