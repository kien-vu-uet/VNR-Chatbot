from typing import Any, List, Tuple, Dict
from sentence_transformers import CrossEncoder
from components.elasticsearch_client import ElasticSearchExecutor
import components.expansion_executor as EE
import numpy as np
import torch
import torch.nn.functional as F
import huggingface_hub
from copy import deepcopy
from tqdm import tqdm

def sigmoid_norm(input, scale, offset):
    return (input ** offset) / ((scale ** offset) + (input ** offset))

def _get_competive_encoder_path(index:str, competitive_matrix:Dict) -> str:
    keys = list(competitive_matrix.keys())
    sorted_keys = sorted(keys, key=lambda x:len(x), reverse=True)
    for key in sorted_keys:
        if key in index:
            return competitive_matrix.get(key)
    return None

class RetrievalExecutor:
    def __init__(self,  model_name_or_path,
                        cache_dir,
                        hf_token,
                        device,
                        batch_size, 
                        max_length,
                        ranking_strategy,
                        max_pool_size,
                        weights,
                        num_workers,
                        expand_pool,
                        # apply_rerank_softmax,
                        **kwargs) -> None:
        huggingface_hub.login(hf_token)
        self.es_executor = ElasticSearchExecutor(device=device, cache_dir=cache_dir, 
                                                 **kwargs['elastic_search'], **kwargs['encoder'])
        self.rerank_model = CrossEncoder(model_name_or_path, device=device, max_length=max_length,
                                        automodel_args={"cache_dir": cache_dir, "token": hf_token})
        self.max_pool_size = max_pool_size
        self.batch_size = batch_size
        self.device = device
        self.ranking_strategy = ranking_strategy
        self.num_workers = num_workers
        self.apply_rerank_softmax = self.rerank_model.config.num_labels >= 2
        self.torch_dtype = torch.float32 if self.device == 'cpu' else torch.float16
        self._expand_pool = getattr(EE, expand_pool['name'])(es_executor=self.es_executor, 
                                                             device=device, 
                                                             cache_dir=cache_dir, 
                                                             **expand_pool['params'])
        self._expand_first = expand_pool['expand_first']
        if weights is None:
            self.weights = torch.tensor([1.0, 1.0, 1.0], dtype=self.torch_dtype, device=self.device).view(-1)
        else:
            self.weights = [weights['bm25'], weights['knn'], weights['rerank']]
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
            "expand_stategy": {
                "executor": self._expand_pool.__dict__,
                "expand_first": self._expand_first
            }
        }
       
    def __prepare_content(self, header, content) -> str:
        sent_separator='<\\>'
        segment_headers, _ = self.es_executor._make_segment_query(header, sent_separator)
        segment_headers = segment_headers.split(sent_separator)
        segment_contents, _ = self.es_executor._make_segment_query(content, sent_separator)
        segment_contents = segment_contents.split(sent_separator)
        segment_headers.reverse()
        for _head in segment_headers:
            if _head not in segment_contents:
                segment_contents.insert(0, _head)
        return ' '.join(segment_contents).replace('_', ' ')
    
    def __call__(self, 
                 question: str, 
                 return_documents: bool = True,
                 cache_explanation: bool = True,
                 **kwargs) -> List[Any]:
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
        final_scores = F.linear(scores, weights) / torch.sum(weights)
        top_scores = final_scores.cpu().tolist()
        indices = range(len(top_scores))
        
        if self.ranking_strategy is not None:
            rank_fn = getattr(self, f"_rank_with_{self.ranking_strategy['name'].lower()}")
            params = self.ranking_strategy['params']
            top_scores, indices = rank_fn(final_scores, **params)
        
        final_outputs = [es_outputs[i].pop('_source') for i in indices]
        
        if return_documents:
            return self.__prepare_documents(final_outputs, top_scores)
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
                             num_workers=0,
                             apply_softmax=self.apply_rerank_softmax,
                             convert_to_numpy=True
                             )
        if outputs.ndim == 1:
            return outputs
        outputs = outputs[:, -1].tolist()
        return outputs
    
    def __prepare_documents(self, pool:List[Dict], scores:List) -> List[Tuple[str, List[float], float]]:
        mapping = {}
        for item, score in zip(pool[:self.max_pool_size], scores[:self.max_pool_size]):
            item['score'] = score
            if item['symbol_number'] in mapping.keys():
                mapping[item['symbol_number']].append(item)
            else:
                mapping[item['symbol_number']] = [item]
           
        final_outputs = []     
        for symbol_number, items in mapping.items():
            item_outputs = []
            items = sorted(items, key=lambda x: x['chunk_index'], reverse=False)
            scores_ = [item['score'] for item in items]
            avg_score = sum(scores_) / len(scores_)
            prefix = f"""METADATA: {items[0]['title']}\n{symbol_number}\n{items[0]['field']}"""
            item_outputs.append(prefix)
            for i in range(len(items)):
                item = items[i]
                item_outputs.append(f"PASSAGE {i+1}: {self.__prepare_content(item['header'], item['content'])}")
            final_outputs.append(('\n'.join(item_outputs), scores_, avg_score))
            
        return final_outputs
    
    def _filter_duplicate(self, pool: List[Dict]) -> List[Dict]:
        exist_id = []
        filtered_pool = []
        for item in pool:
            if item['_id'] not in exist_id:
                filtered_pool.append(item)
                exist_id.append(item['_id'])
        return filtered_pool
    
    def rerank_with_expanded_query(self, 
                 question: str, 
                 return_documents: bool = True,                     
                 **kwarg) -> List[Any]:
        es_outputs = self.es_executor(question)
        if self._expand_first:
            es_outputs.extend(self._expand_pool(query=question, pool=es_outputs))
            es_outputs = self._filter_duplicate(es_outputs)
        es_scores = []
        documents = []
        for hit in es_outputs:
            documents.append(self.__prepare_content(hit['_source']['header'], hit['_source']['content']))
            es_scores.append(hit['_score'])
                
        rerank_scores = self._rerank_inference(question, documents)
        scores = [item + [r_score] for item, r_score in zip(es_scores, rerank_scores)]
        scores = torch.tensor(scores, dtype=self.torch_dtype, device=self.device)
        weights = self.weights[:scores.shape[-1]]
        final_scores = F.linear(scores, weights) / torch.sum(weights)
        top_scores = final_scores.cpu().tolist()
        indices = range(len(top_scores))
        
        if self.ranking_strategy is not None:
            rank_fn = getattr(self, f"_rank_with_{self.ranking_strategy['name'].lower()}")
            params = self.ranking_strategy['params']
            top_scores, indices = rank_fn(final_scores, **params)
            
        final_outputs = [es_outputs[i] for i in indices]
        
        if not self._expand_first:
            rerank_outputs = [es_outputs[i] for i in indices]
            final_outputs.extend(self._expand_pool(query=question, pool=rerank_outputs))
            final_outputs = self._filter_duplicate(final_outputs)
        
        final_outputs = [item.pop('_source') for item in final_outputs]
        if return_documents:
            return self.__prepare_documents(final_outputs, top_scores)
        return top_scores