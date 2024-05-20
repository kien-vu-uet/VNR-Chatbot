from components.elasticsearch_client import ElasticSearchExecutor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import huggingface_hub
from typing import Dict, List
from copy import deepcopy
from tqdm import tqdm
from gradio_client import Client
import os, requests
from components import FIELD as field_list

class BaseExpansion:
    def __init__(self, es_executor: ElasticSearchExecutor,
                 num_steps:1,
                 **kwargs) -> None:
        self.es_executor = es_executor
        self.num_steps = num_steps
        
    def __call__(self, **kwargs) -> List[Dict]:
        pass
        
class KNeighborsExpansion(BaseExpansion):
    def __init__(self, es_executor: ElasticSearchExecutor, 
                 num_steps: int,
                 same_header: bool,
                 **kwargs) -> None:
        super().__init__(es_executor, num_steps, **kwargs)
        self.same_header = same_header
    
    @property
    def __dict__(self):
        return {
            "name": str(self.__class__),
            "num_steps": self.num_steps,
            "same_header": self.same_header
        }
    
    def __call__(self, pool: List[Dict], **kwargs) -> List[Dict]:
        basic_query = []        
        if 'bm25' in self.es_executor._last_query.keys():
            bm25_query = deepcopy(self.es_executor._last_query['bm25'])
            # basic_query.extend(bm25_query['function_score']['query']['bool']['should'])
            basic_query.append(bm25_query)
            
        if 'knn' in self.es_executor._last_query.keys():
            knn_query = deepcopy(self.es_executor._last_query['knn'])
            if 'k' in knn_query.keys(): knn_query.pop('k')
            basic_query.append({'knn': knn_query})
        
        index_mapping = {}
        header_mapping = {}

        for item in pool:
            item['_source'].pop('content_vector')
            if item['_source']['symbol_number'] in index_mapping.keys():
                index_mapping[item['_source']['symbol_number']].append(item['_source']['chunk_index'])
            else:
                index_mapping[item['_source']['symbol_number']] = [item['_source']['chunk_index']]
                
            header_mapping[item['_source']['symbol_number']] = item['_source']['header']
        
        expand_index_mapping = {k:[] for k,_ in index_mapping.items()}
        
        for symbol_number, chunk_indexs in index_mapping.items():
            for idx in chunk_indexs:
                expand_idx = [i for i in range(idx-self.num_steps, idx+self.num_steps+1) \
                                if i not in chunk_indexs and i >= 0]
                expand_index_mapping[symbol_number].extend(expand_idx)
            expand_index_mapping[symbol_number] = list(set(expand_index_mapping[symbol_number]))
            
        expand_query = {
            "bool": {
                "should": []
            }
        }    
        for symbol_number, chunk_indexs in expand_index_mapping.items():
            query_item = [
                {"match_phrase": {"symbol_number": symbol_number}},
                {"terms": {"chunk_index": chunk_indexs}},
            ]
            if self.same_header:
                query_item.append({"match_phrase": {"header": header_mapping[symbol_number]}})
            
            expand_query['bool']['should'].append(
                {
                    "bool": {
                        "must": [
                            *query_item
                        ],
                        "should": [
                            *basic_query
                        ]
                    }
                }
            )
            
        response = self.es_executor.client.search(
                                                index=self.es_executor.index,
                                                query=expand_query,
                                                explain=True)
        
        expand_outputs = []
        for item in tqdm(response['hits']['hits'], desc='Rescore'):
            bm25_score = 0
            knn_score = 0
            if item['_explanation']['details'][0]['details'][-1]['description'].startswith('min of:'):
                bm25_score = item['_explanation']['details'][0]['details'][-1]['value']
            elif item['_explanation']['details'][0]['details'][-1]['description'].startswith('within top'):
                bm25_score = item['_explanation']['details'][0]['details'][-2]['value']
                knn_score = item['_explanation']['details'][0]['details'][-1]['value']
            item['_score'] = [bm25_score, self.es_executor._normalized_cosine(knn_score)]
            item.pop('_explanation')
            expand_outputs.append(item)

        # expand_outputs.extend(pool)   
        return expand_outputs

class Doc2QueryExpansion(BaseExpansion):
    def __init__(self, es_executor: ElasticSearchExecutor, 
                 num_steps: int,
                 model_name_or_path, 
                 cache_dir,
                 device,
                 llm_api,
                **kwargs) -> None:
        super().__init__(es_executor, num_steps, **kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path,
                                                           cache_dir=cache_dir).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, 
                                                       cache_dir=cache_dir)
        username = os.getenv('USERNAME')
        password = os.getenv('PASSWORD')
        # self.llm_client = Client(llm_api, auth=(username, password))
        self.llm_client = llm_api
        self.device = device
        self.generate_kwargs = kwargs['generate_kwargs'] if 'generate_kwargs' in kwargs.keys() \
                                else dict(
                                        max_length=128,
                                        do_sample=True,
                                        top_p=0.9,
                                        top_k=10, 
                                        # no_repeat_ngram_size=2,
                                    )
    
    @property
    def __dict__(self):
        return {
            "name": str(self.__class__),
            "num_steps": self.num_steps,
            "doc2query": self.model.config._name_or_path,
            "generate_kwargs": self.generate_kwargs
        }
       
    def __call__(self, query:str, **kwargs) -> List[Dict]:
        headers = {
            'Content-Type': 'application/json',
            'Authorization':  f"Bearer {os.getenv('LLM_SERVER_API_KEY')}"
        }
        extend_outputs = []
        for i in tqdm(range(self.num_steps)):
            prompt = f"[INST]  <<SYS>> \nBạn là một trợ lý tư vấn pháp lý về các vấn đề Luật và Quy chuẩn - Tiêu chuẩn. "\
                     f"Hãy suy luận để phản hồi các yêu cầu của người dùng dựa trên tài liệu hoặc thông tin được cung cấp (nếu có). " \
                     f"Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. "\
                     f"Nếu bạn không biết câu trả lời cho một câu hỏi hoặc không tìm thấy thông tin liên quan, hãy trả lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch. <</SYS>> \n\n"\
                     f"Từ kết quả phân tích, hãy viết 1 đoạn ngắn để trả lời cho câu hỏi \"{query}\"! [/INST]"
                        
                    #  f"Trước hết, bạn hãy phân tích câu hỏi \"{query}\" " \
                    #  f"thuộc lĩnh vực/loại hình công việc nào trong danh sách sau:\n{field_list}\n" \
            
            json_data = {
                "prompt": prompt, 
                "n_predict": 256,
                "seed": -1,
                "temp": 0.3,
                "top_p": 0.9,
                "top_k": 100,
                "logit_bias": os.getenv('LOGIT_BIAS'),
                "repeat_penalty": 1.05,
                "repeat_last_n": 128,
            }
            response = requests.post(self.llm_client, 
                             headers=headers, 
                             json=json_data)
            hypo_doc = response.json().get('content')
            new_query = self._doc2query_inference(hypo_doc)
            extend_outputs.extend(self.es_executor(query + '\n' + new_query))
            query = new_query
            print(hypo_doc)
            print(query)
        return extend_outputs
            
    def _doc2query_inference(self, para:str) -> str:
        input_ids = self.tokenizer.encode(para, return_tensors='pt').to(self.device)
        with torch.no_grad():
            # Here we use top_k / top_k random sampling. It generates more diverse queries, but of lower quality
            output_ids = self.model.generate(
                                    input_ids=input_ids,
                                    # num_beams=1, 
                                    num_return_sequences=1,
                                    **self.generate_kwargs
                                )
            
            new_query = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            del input_ids
            del output_ids
            return new_query
            
            # Here we use Beam-search. It generates better quality queries, but with less diversity
            # beam_outputs = self.model.generate(
            #     input_ids=input_ids, 
            #     max_length=128, 
            #     num_beams=5, 
            #     no_repeat_ngram_size=2, 
            #     num_return_sequences=5, 
            #     early_stopping=True
            # )
        
        
kNE = KNeighborsExpansion
D2QE = Doc2QueryExpansion