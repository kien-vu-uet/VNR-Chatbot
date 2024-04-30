from typing import Any, List, Tuple, Dict
from sentence_transformers import CrossEncoder
from components.elasticsearch_client import ElasticSearchExecutor
import numpy as np
import torch
import torch.nn.functional as F
import huggingface_hub
from copy import deepcopy
from tqdm import tqdm

def sigmoid_norm(input, scale, offset):
    return (input ** offset) / ((scale ** offset) + (input ** offset))

class RetrievalExecutor:
    def __init__(self,  model_name_or_path,
                        cache_dir,
                        hf_token,
                        device,
                        batch_size, 
                        max_length,
                        ranking_strategy,
                        weights,
                        num_workers,
                        apply_rerank_softmax,
                        **kwargs) -> None:
        self.es_executor = ElasticSearchExecutor(device=device, cache_dir=cache_dir, 
                                                 **kwargs['elastic_search'], **kwargs['encoder'])
        self.rerank_model = CrossEncoder(model_name_or_path, device=device, max_length=max_length,
                                        automodel_args={"cache_dir": cache_dir, "token": hf_token})
        huggingface_hub.login(hf_token)
        self.batch_size = batch_size
        self.device = device
        self.ranking_strategy = ranking_strategy
        self.num_workers = num_workers
        self.apply_rerank_softmax = apply_rerank_softmax
        self.torch_dtype = torch.float32 if self.device == 'cpu' else torch.float16
        if weights is None:
            self.weights = torch.tensor([1.0, 1.0], dtype=self.torch_dtype, device=self.device).view(-1)
        else:
            self.weights = [v for k,v in weights.items()]
            self.weights = torch.tensor(self.weights, dtype=self.torch_dtype, device=self.device).view(-1)
            
    @property
    def __dict__(self):
        return {
            "es_executor": self.es_executor.__dict__,
            "rerank_model": self.rerank_model.config.name_or_path,
            "batch_size": self.batch_size,
            "device": self.device,
            "ranking_strategy": self.ranking_strategy,
            "num_workers": self.num_workers,
            "apply_rerank_softmax": self.apply_rerank_softmax,
            "torch_dtype": self.torch_dtype.__str__(),
            "weights": self.weights.tolist(),
        }
       
    def __prepare_content(self, header, content) -> str:
        sent_separator='<\\>'
        segment_headers = self.es_executor._make_segment_query(header, sent_separator).split(sent_separator)
        segment_contents = self.es_executor._make_segment_query(content, sent_separator).split(sent_separator)
        segment_headers.reverse()
        for _head in segment_headers:
            if _head not in segment_contents:
                segment_contents.insert(0, _head)
        return ' '.join(segment_contents).replace('_', ' ')
    
    def __call__(self, 
                 question: str, 
                 return_documents: bool = True,
                 ) -> List[Any]:
        es_outputs = self.es_executor(question)
        es_scores = []
        documents = []
        for hit in es_outputs:
            documents.append(self.__prepare_content(hit['_source']['header'], hit['_source']['content']))
            es_scores.append(hit['_score'])
                
        rerank_scores = self._rerank_inference(question, documents)
        scores = [item + [r_score] for item, r_score in zip(es_scores, rerank_scores)]
        scores = torch.tensor(scores, dtype=self.torch_dtype, device=self.device)
        weights = self.weights[:scores.shape[-1]]
        final_scores = F.linear(scores, weights) / (torch.sum(weights) + weights[0])
        top_scores = final_scores.cpu().tolist()
        indices = range(len(top_scores))
        
        if self.ranking_strategy is not None:
            rank_fn = getattr(self, f"_rank_with_{self.ranking_strategy['name'].lower()}")
            params = self.ranking_strategy['params']
            top_scores, indices = rank_fn(final_scores, **params)
        
        final_outputs = [es_outputs[i].pop('_source') for i in indices]
        
        if return_documents:
            return list(zip(self.__prepare_documents(final_outputs), top_scores))
        return top_scores
        
    @staticmethod
    def _rank_with_topk(scores: torch.Tensor, topk:int, relative:bool,
                        **kwargs) -> Tuple[List[int], List[int]]:
        if relative:
            if topk > 100: topk = 100
            topk = round((topk / 100) * scores.shape[0])
        else:
            if topk > scores.shape[0]:
                topk = scores.shape[0]
        values, indices = torch.topk(scores.view(-1), k=topk, dim=-1, largest=True, sorted=True)
        return values.cpu().tolist(), indices.cpu().tolist()
    
    @staticmethod
    def _rank_with_threshold(scores: torch.Tensor, 
                             confidence_threshold: float, 
                             ambiguity_threshold: float,
                             **kwargs
                             ) -> Tuple[List[int], List[int]]:
        sorted_scores, indices = torch.sort(scores.view(-1), dim=-1, descending=True)
        ambiguity_scores = sorted_scores[:-1] - torch.clone(sorted_scores)[1:]
        sorted_scores = sorted_scores.cpu().tolist()
        indices = indices.cpu().tolist()
        ambiguity_scores = [0.] + ambiguity_scores.cpu().tolist()
        out_scores, out_indices = [], []
        for _score, _index, _ambi_score in zip(sorted_scores, indices, ambiguity_scores):
            if _score >= confidence_threshold and _ambi_score <= ambiguity_threshold:
                out_scores.append(_score)
                out_indices.append(_index)
            else: break
                
        return out_scores, out_indices

    def _rerank_inference(self, question:str, documents:List[str]) -> List[int]:
        inputs = [[question, document] for document in documents]
        outputs = self.rerank_model.predict(inputs, 
                             batch_size=self.batch_size,
                             show_progress_bar=True,
                             num_workers=self.num_workers,
                             apply_softmax=self.apply_rerank_softmax,
                             convert_to_numpy=True
                             )
        if outputs.ndim == 1:
            return outputs
        outputs = outputs[:, -1].tolist()
        return outputs
    
    def __prepare_documents(self, pool:Dict) -> List[str]:
        mapping = {}
        for item in pool:
            if item['symbol_number'] in mapping.keys():
                mapping[item['symbol_number']].append(item)
            else:
                mapping[item['symbol_number']] = [item]
           
        final_outputs = []     
        for symbol_number, items in mapping.items():
            item_outputs = []
            prefix = f"""{items[0]['title']}\n{symbol_number}\n{items[0]['field']}"""
            item_outputs.append(prefix)
            for item in items:
                item_outputs.append(self.prepare_content(item['header'], item['content']))
            final_outputs.append('\n'.join(item_outputs))
            
        return final_outputs
    
    def rerank_with_expanded_query(self, 
                 question: str, 
                 num_steps: int, 
                 same_header: bool,
                 expand_first: bool,
                 return_documents: bool = True,                     
                 **kwarg) -> List[Any]:
        es_outputs = self.es_executor(question)
        if expand_first:
            es_outputs = self._expand_pool(es_outputs, num_steps, same_header)
        es_scores = []
        documents = []
        for hit in es_outputs:
            documents.append(self.__prepare_content(hit['_source']['header'], hit['_source']['content']))
            es_scores.append(hit['_score'])
                
        rerank_scores = self._rerank_inference(question, documents)
        scores = [item + [r_score] for item, r_score in zip(es_scores, rerank_scores)]
        scores = torch.tensor(scores, dtype=self.torch_dtype, device=self.device)
        weights = self.weights[:scores.shape[-1]]
        final_scores = F.linear(scores, weights) / (torch.sum(weights) + weights[0])
        top_scores = final_scores.cpu().tolist()
        indices = range(len(top_scores))
        
        if self.ranking_strategy is not None:
            rank_fn = getattr(self, f"_rank_with_{self.ranking_strategy['name'].lower()}")
            params = self.ranking_strategy['params']
            top_scores, indices = rank_fn(final_scores, **params)
            
        final_outputs = [es_outputs[i] for i in indices]
        
        if not expand_first:
            rerank_outputs = [es_outputs[i] for i in indices]
            final_outputs = self._expand_pool(rerank_outputs, num_steps, same_header)
        
        final_outputs = [item.pop('_source') for item in final_outputs]
        if return_documents:
            return list(zip(self.__prepare_documents(final_outputs), top_scores))
        return top_scores
    
    def _expand_pool(self, pool: Dict, 
                    num_steps: int, 
                    same_header: bool,
                    **kwargs) -> Dict:
        basic_query = []        
        if 'bm25' in self.es_executor._last_query.keys():
            bm25_query = deepcopy(self.es_executor._last_query['bm25'])
            # basic_query.extend(bm25_query['function_score']['query']['bool']['should'])
            basic_query.append(bm25_query)
            
        if 'knn' in self.es_executor._last_query.keys():
            knn_query = deepcopy(self.es_executor._last_query['knn'])
            knn_query.pop('k')
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
                expand_idx = [i for i in range(idx-num_steps, idx+num_steps+1) \
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
            if same_header:
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
            item['_score'] = [bm25_score, knn_score]
            item.pop('_explanation')
            expand_outputs.append(item)

        expand_outputs.extend(pool)   
        return expand_outputs