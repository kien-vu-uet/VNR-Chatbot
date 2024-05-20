from fastapi import FastAPI
import gradio as gr
from app import demo
import os

app = FastAPI()

username = os.getenv('USERNAME')
password = os.getenv('PASSWORD')

app = gr.mount_gradio_app(app, 
                          demo.queue(max_size=10), 
                          path='/api', 
                          auth=(username, password),
                          app_kwargs={"docs_url": "/docs"},
                          favicon_path='./favicon.png')

@app.get('/')
async def root():
    return 'Gradio app is running at /api', 200


# uvicorn api:app --host=0.0.0.0 --port=9669 --app-dir=.