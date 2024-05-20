import gradio as gr
import yaml, json
from components.retrieval_executor import RetrievalExecutor, _get_competive_encoder_path
from typing import Dict, List, Any
import os
import torch
from copy import deepcopy
from collections import OrderedDict

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

global competitive_matrix, competitive_path
competitive_path = os.path.join(BASE_DIR, 'config/competitive_matrix.json')
competitive_matrix = json.load(open(competitive_path, 'r'))

config_path = os.path.join(BASE_DIR, 'config/last_config.yml') \
                       if os.path.exists(os.path.join(BASE_DIR, 'config/last_config.yml')) \
        else os.path.join(BASE_DIR, 'config/full_combo_topk.yml')


with open(config_path, 'r') as f:
    opt = yaml.safe_load(f)
    opt['encoder']['encoder_name_or_path'] = _get_competive_encoder_path(opt['elastic_search']['index'], competitive_matrix)
    f.close()

global retrieval
retrieval = RetrievalExecutor(**opt, **opt['rerank_model']) 
    
def flatten_dict(_dict:Dict, prefix=[], index=1) -> List[List[Any]]:
    result = []
    for k,v in _dict.items():
        if k == 'encoder_name_or_path': continue
        prefix_ = prefix + [k]
        if isinstance(v, Dict):
            result.extend(flatten_dict(_dict[k], prefix_, index+len(result)))
        else:
            result.append([index+len(result), '.'.join(prefix_),  str(v)])
    return [r for r in result if len(r) == 3 and len(''.join(r[1:])) > 0]
    
def reload_config_2(args:List[List[Any]]):    
    def fill_value(_dict: Dict, keys:List[str], value:str) -> Dict:
        if len(keys) == 1: 
            try:
                _dict[keys[0]] = eval(value)
            except:
                _dict[keys[0]] = value
            return _dict
        
        key = keys.pop(0)
        if key not in _dict:
            _dict[key] = {}
        _dict[key] = fill_value(_dict[key], keys, value)
        return _dict
    
    try:
        global retrieval, opt
        opt_ = {}
        for _, keys, value in args:
            keys = keys.split('.')
            opt_ = fill_value(opt_, keys, value)
        if opt_ != opt:
            opt = opt_
            opt['encoder']['encoder_name_or_path'] = _get_competive_encoder_path(opt['elastic_search']['index'], competitive_matrix)
            print(opt['encoder']['encoder_name_or_path'])
            torch.cuda.empty_cache()
            retrieval = RetrievalExecutor(**opt, **opt['rerank_model'])
            yaml.safe_dump(opt, open(os.path.join(BASE_DIR, 'config/last_config.yml'), 'w'), indent=2, sort_keys=False)
        return retrieval.__dict__
    except Exception as e:
        raise Exception(f"Cannot load configuration! Got {e.args}!")
    

def search(query: str, expand_query:bool) -> Dict:
    try:
        global retrieval, opt
        outputs = retrieval(query, return_documents=True) if not expand_query \
             else retrieval.rerank_with_expanded_query(query, return_documents=True)
        final_outputs = []
        for _doc, _scores, _avg_score in outputs:
            final_outputs.append(
                {
                    "_avg_score": _avg_score,
                    "_doc": _doc,
                    "_score": _scores,
                }
            )
        return final_outputs
    except Exception as e:
        raise Exception(f"Cannot serving request! Got {e.args}!")
        
with gr.Blocks(css='style.css', title='RA-Backend') as demo:
    with gr.Tab(label='Config'):
        flat_dict = flatten_dict(opt)
        
        input_component = gr.DataFrame(
                            label='Arguments',
                            headers=['Ord', 'Key', 'Value'],
                            datatype=['number', 'str', 'str'],
                            row_count=len(flat_dict),
                            col_count=(3, 'fixed'),
                            value=flat_dict,
                            interactive=True,
                            type='array',
                            height=700,
                            # min_width=650
                        )
        # input_component.change(fn=None)
        # input_component.select(fn=None)
        # input_component.input(fn=None)
        sb_bttn = gr.Button(value='Submit', interactive=True)
        output_component = gr.JSON(value=retrieval.__dict__, label='Configuration')
        
        gr.on(triggers=[sb_bttn.click],
               fn=reload_config_2,
               inputs=input_component,
               outputs=output_component,
               trigger_mode='always_last',
               api_name='config')
        
    with gr.Tab(label='Search'):
        tab2 = gr.Interface(fn=search,
                     inputs=['text'],
                     outputs='json',
                     api_name='search', 
                     examples=[
                        ['Quy định về đăng kiểm giao thông đường thuỷ mới nhất?'],
                        ['Những nguyên tắc cơ bản của giao thông đường sắt là gì?']
                    ],
                    #  _api_mode=True,
                     additional_inputs=gr.Checkbox(value=True, label='Expand queries'),
                     allow_flagging='auto',
                     stop_btn='Stop'
                     )
        with gr.Accordion(label="Last query", visible=True, open=False) as view:
            view_bttn = gr.Button('View', size='sm', interactive=True)
            last_bm25_query = gr.JSON(label='BM25')
            @gr.on(triggers=[view_bttn.click],
                   outputs=last_bm25_query)
            def get_bm25():
                return retrieval.es_executor._last_query['bm25']
        
    with gr.Tab(label='Competitve Matrix'):
        matrix = gr.DataFrame(
                    headers=['Index Keyword', 'Encoder Name or Path'],
                    datatype=['str', 'str'],
                    row_count=len(competitive_matrix),
                    col_count=(2, 'fixed'),
                    value=[list(item) for item in competitive_matrix.items()],
                    interactive=True,
                    type='array'
                )
        submit_bttn = gr.Button(value='Submit')
        @gr.on(triggers=[submit_bttn.click],
               inputs=matrix,
               trigger_mode='always_last')
        def save_matrix(args:List[List[str]]) -> None:
            for k, v in args:
                competitive_matrix.update({k:v})
            json.dump(competitive_matrix, open(competitive_path, 'w'))
            raise