from fastapi import FastAPI
import gradio as gr
import os

from app import demo

app = FastAPI()

username = os.getenv('USERNAME')
password = os.getenv('PASSWORD')

@app.get('/')
async def root():
    return 'Gradio app is running at /chat', 200

app = gr.mount_gradio_app(app, 
                          demo.queue(max_size=50), 
                          path='/chat',
                        #   auth=(username, password),
                          app_kwargs={"docs_url": "/docs"},
                          )