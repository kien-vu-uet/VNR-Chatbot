from elasticsearch import Elasticsearch
import os
from sentence_transformers import SentenceTransformer
import py_vncorenlp


ELASTIC_API_KEY = os.getenv('ELASTIC_API_KEY', 'WnpxRjc0MEJ0OUFnM2g4V216amE6STh3aDNnd1lTd1dGdS1FMER5enVOUQ==')
client = Elasticsearch('http://192.168.48.2:9200', api_key=ELASTIC_API_KEY)
model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', cache_folder='/hf_cache').to('cuda:0')
vncorenlp_segmentor = py_vncorenlp.VnCoreNLP(save_dir='/vncorenlp')

def query(question:str):
    query = {
        "bool": {
            "match": {
                "content": question
            }
        }
    }
    knn = {
        "field": "title_vector",
        "query_vector": model.encode(
            ' '.join(vncorenlp_segmentor.word_segment(question))
        ).tolist(),  # generate embedding for query so it can be compared to `title_vector`
        "k": 20,
        "num_candidates": 10,
    }
    rank={"rrf": {}}
    
    response = client.search(index='test-index', query=query, knn=knn)
    
    for hit in response['hits']['hits']:
        print(hit['_source'])
