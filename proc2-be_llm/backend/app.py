import gradio as gr
import yaml, json
from components.retrieval_executor import RetrievalExecutor
from typing import Dict, List
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

path = os.path.join(BASE_DIR, 'config/last_config.yml') \
                       if os.path.exists(os.path.join(BASE_DIR, 'config/last_config.yml')) \
        else os.path.join(BASE_DIR, 'config/full_combo_topk.yml')

with open(path, 'r') as f:
    global opt
    opt = yaml.safe_load(f)
    f.close()

global retrieval
retrieval = RetrievalExecutor(**opt, **opt['rerank_model']) 

def reload_config(new_config:str) -> Dict:
    try:
        global retrieval, opt
        opt_ = yaml.safe_load(new_config)
        if opt_ != opt:
            opt = opt_
            retrieval = RetrievalExecutor(**opt, **opt['rerank_model'])
        yaml.safe_dump(opt, open('./config/last_config.yml', 'w'), indent=2, sort_keys=False)
        return retrieval.__dict__
    except Exception as e:
        raise Exception(f"Cannot load configuration! Got {e.args}!")

def search(query: str, expand_query:bool) -> Dict:
    try:
        global retrieval, opt
        outputs = retrieval(query, return_documents=True) if not expand_query \
             else retrieval.rerank_with_expanded_query(query,return_documents=True,
                                                       **opt['expand_pool'])
        final_outputs = []
        for _doc, _score in outputs:
            final_outputs.append(
                {
                    "_score": _score,
                    "_doc": _doc,
                }
            )
        return final_outputs
    except Exception as e:
        raise Exception(f"Cannot serving request! Got {e.args}!")
        
with gr.Blocks(css='style.css', title='VNR-Chatbot') as demo:
    with gr.Tab(label='Config'):
        tab1 = gr.Interface(fn=reload_config, 
                     inputs='text', 
                     outputs='json', 
                     examples=[yaml.safe_dump(opt, indent=6, sort_keys=False)],
                     cache_examples=True,
                     api_name='config',
                    #  _api_mode=True,
                     allow_flagging='auto',
                     )

        for component in tab1.input_components:
            if isinstance(component, gr.Textbox):
                component.value = yaml.safe_dump(opt, indent=6, sort_keys=False)
                component.max_lines = 50
                component.lines = 50
                break
        
        for component in tab1.output_components:
            if isinstance(component, gr.JSON):
                component.value = retrieval.__dict__
                break
            
            if isinstance(component, gr.Examples) or isinstance(component.parent, gr.Examples):
                component.visible = False
                break
        
    with gr.Tab(label='Search'):
        tab2 = gr.Interface(fn=search,
                     inputs=['text'],
                     outputs='json',
                     api_name='search', 
                    #  _api_mode=True,
                     additional_inputs=gr.Checkbox(value=False, label='Expand queries'),
                     allow_flagging='auto',
                     )