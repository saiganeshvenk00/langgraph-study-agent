import os
import uuid
import gradio as gr
from dotenv import load_dotenv
from graph import run_graph



session_id = str(uuid.uuid4())

def chat(message, history):
    return run_graph(message, session_id)

demo = gr.ChatInterface(
    fn=chat, #other available functions are: chatbot, chatbot_stream, chatbot_stream_mode, chatbot_stream_mode_v2
    title="Study-Buddy",
   #type="messages" #other available types are: "messages", "text", "single", "file", "image", "audio", "video", "file", "image", "audio", "video"
    #"messages" is the default type
    #"text" is for single text input
    #"single" is for single input
    #"file" is for file input
    #"image" is for image input
    #"audio" is for audio input
    #"video" is for video input
)

#syntax for gradio chat interface is:
'''gr.ChatInterface(
    fn=chat, 
    title="Study Helper",
    type="messages" 
)'''

if __name__ == "__main__":
    demo.launch()

