from elasticsearch import Elasticsearch
import os, json
from sentence_transformers import SentenceTransformer
import py_vncorenlp
import torch
from typing import Dict, List, Tuple
import requests
from copy import deepcopy


class ElasticSearchExecutor:
    def __init__(self, 
                 host,
                 token, 
                 device,
                 encoder_name_or_path,
                 segment_api,
                 cache_dir,
                 index, 
                 query_type="should",
                 activate_fn="sigmoid",
                 scale=1,
                 offset=0.5,
                 knn_topk=20,
                 knn_num_candidates=100,
                 query_strategy="ensemble",
                 **kwargs) -> None:
        self.client = Elasticsearch(host, api_key=token)
        print(self.client.info())
        self.device = device
        # encoder_name_or_path = _get_competive_encoder_path(index_name=index)
        self.encoder = SentenceTransformer(encoder_name_or_path, cache_folder=cache_dir, device=device)
        # self.vncorenlp_segmentor = py_vncorenlp.VnCoreNLP(annotators=['wseg'], 
        #                                                   max_heap_size=max_heap_size,
        #                                                   save_dir=vncorenlp_path)
        self.index = index
        self.query_type = query_type
        self.activate_fn = activate_fn
        self.scale = scale
        self.offset = offset
        self.knn_topk = knn_topk
        self.knn_num_candidates = knn_num_candidates
        self.query_strategy = query_strategy
        self.segment_api = segment_api
        assert self.query_type in ["should", "must", "must_not"], "Got unexpected query type!"
        assert self.activate_fn in ["sigmoid", "softmax"], "Got unexpected activate function!"
        self.fn_source = f"sigmoid(_score, {self.scale}, {self.offset})" if self.activate_fn == "sigmoid" else \
                         f"softmax(_score, {self.scale})"    
        self._last_query = {"bm25": {}, "knn": {}}
        
    @property
    def __dict__(self):
        return {
            "client": json.dumps(dict(self.client.info()), indent=4, sort_keys=False),
            "device": self.device,
            "encoder": self.encoder._first_module().tokenizer.name_or_path,
            "index": self.index,
            "query_type": self.query_type,
            "activate_fn": {
                "name" : self.activate_fn,
                "scale": self.scale,
                "offset": self.offset,
            },
            "knn": {
                "topk": self.knn_topk,
                "num_candidates": self.knn_num_candidates
            },
            "query_strategy": self.query_strategy,
            "segment_api": self.segment_api,
        }
        
    @staticmethod
    def _normalized_cosine(_score:float) -> float:
        return (_score + 1.0) / 2.0  
    
    def __call__(self, question:str) -> List[Dict]: # query_type='should', fn='sigmoid', scale=10, offset=0.5, knn_topk=20, knn_num_candidates=100):
        # rank={"rrf": {}}
        question_segment, handled_question_segment = self._make_segment_query(question)
        bm25_query = self._make_bm25_query(question, handled_question_segment)
        if self.query_strategy == 'bm25_only':
            response = self.client.search(index=self.index, 
                                          query=bm25_query, 
                                         )
            outputs = response['hits']['hits']
            for item in outputs:
                item['_score'] = [item['_score'], 0.]
            return outputs
        else:
            knn_query, script_fields = self._make_knn_query(question_segment)
            if self.query_strategy == 'semantic':
                response = self.client.search(index=self.index, 
                                            knn=knn_query, 
                                            #   script_fields=script_fields,
                                            )
                outputs = response['hits']['hits']
                for item in outputs:
                    knn_score = item['_score']
                    item['_score'] = [0., self._normalized_cosine(knn_score)]
                return outputs
            elif self.query_strategy == 'ensemble':
                response = self.client.search(index=self.index, 
                                            query=bm25_query, 
                                            knn=knn_query, 
                                            explain=True
                                            #   script_fields=script_fields,
                                            )
                
                outputs = response['hits']['hits']
                for item in outputs:
                    bm25_score = item['_explanation']['details'][0]['value']
                    knn_score = item['_explanation']['details'][-1]['value']
                    item['_score'] = [bm25_score, self._normalized_cosine(knn_score)]
                    item.pop('_explanation')
                return outputs
            else:
                raise Exception(f"Cannot implement {self.query_strategy}!")
            
            
    def _make_bm25_query(self, question, question_segment:str=None) -> Dict:
        if not question_segment:
            _, question_segment = self._make_segment_query(question)
            
        self._last_query["bm25"] = {
            "function_score": {
                "query": {
                    "bool": {
                        self.query_type: [
                            {
                                "multi_match": {
                                    "query": question,
                                    "fields": [
                                        "content",
                                        "header",
                                        "description",
                                        "title",
                                        "field",
                                    ],
                                    "operator": "or",
                                    "type": "most_fields"
                                }
                            },
                            {
                                "multi_match": {
                                    "query": question_segment,
                                    "fields": [
                                        "header_segment",
                                        "content_segment"
                                    ],
                                    "operator": "or",
                                    "type": "most_fields"
                                }
                            }
                        ]
                    }
                },
                "boost": 1,
                "boost_mode": "replace",
                "functions": [
                    {
                        "script_score": {
                            "script": {
                                "source": self.fn_source
                            }
                        }
                    }
                ]
            }
        }
        return self._last_query['bm25']
        
        
    def _make_segment_query(self, question, 
                            sent_separator:str=' ', 
                            handle_phrase:List[Tuple[str, str]]=[('_', '')]
                            ) -> Tuple[str, str]:
        response = requests.post(self.segment_api['host'], 
                                         json={self.segment_api['input_field']: question, 'sent_separator': sent_separator})
        output = response.json().get(self.segment_api['output_field'])
        handled_output = deepcopy(output).split(' ')
        for char_delimiter, char_replace in handle_phrase:
            handled_phrases = [phrase.replace(char_delimiter, char_replace) for phrase in handled_output if char_delimiter in phrase]
            handled_phrases = [phrase for phrase in handled_phrases if phrase not in handled_output]
            handled_output.extend(handled_phrases)
        return output, ' '.join(handled_output)
        
        
    def _make_knn_query(self, question_segment) -> Dict:
        self._last_query['knn'] = {
            "field": "content_vector",
            "query_vector": self.encoder.encode(question_segment).tolist(),  
            "k": self.knn_topk,
            "num_candidates": self.knn_num_candidates,
            # "score_mode": "script",
            # "params": {
            #     "max_score": 1.0,
            #     "min_score": 0.0
            # }
        }
        self._last_query['script_fields'] = {
            "cosine_normalized_score": {
                "script": {
                    "source": "(params.max_score - params.min_score) * (1 + _score) / 2 + params.min_score",
                    "params": {
                        "max_score": 1.0,
                        "min_score": 0.0
                    }
                }
            }
        }
        return self._last_query['knn'], self._last_query['script_fields']