from fastapi import FastAPI
import gradio as gr

from app import demo

app = FastAPI()

@app.get('/')
async def root():
    return 'Gradio app is running at /chat', 200

app = gr.mount_gradio_app(app, demo.queue(max_size=50), path='/chat')