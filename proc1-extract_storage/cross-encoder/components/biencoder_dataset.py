from typing import List
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample
import json, os

class TripletDataset(SentencesDataset):
    def __init__(self,
                 data_path,
                 model,
                 tokenizer=None,
                 processed_data=None,
                 **kwargs) -> None:
        raw_data = json.load(open(data_path, 'r')) if data_path else []
        self.tokenizer = tokenizer
        self.model = model
        self.data = []
        if processed_data:
            self.data = processed_data
        for item in raw_data:
            try:
                anchor = item['anchor'].replace('_', ' ')
                pos = item['pos'].replace('_', ' ')
                hard_neg = item['hard_neg'].replace('_', ' ')   
                triplet_item = {
                    "texts": [
                        self.tokenizer(anchor),
                        self.tokenizer(pos),
                        self.tokenizer(hard_neg)
                    ]
                }
                self.data.append(triplet_item)
            except Exception as e:
                print(e.args)
        super().__init__(self.input_examples, model)
        
    @property
    def input_examples(self):
        return [InputExample(**item_) for item_ in self.data]
        
    def save_to_disk(self, json_path, indices=None, **kwargs):
        saved_data = self.data if indices is None else [self.data[idx] for idx in indices]
        if not os.path.exists(os.path.dirname(json_path)): 
            os.mkdir(os.path.dirname(json_path))
        json.dump(saved_data, open(json_path, 'w'))
        
    @classmethod
    def load_from_disk(cls, json_path, model, **kwargs):
        data = json.load(open(json_path, 'r'))

        return cls(data_path=None,
                   processed_data=data,
                   model=model)
    
class PairDataset(TripletDataset):
    def __init__(self, 
                 data_path,
                 model,
                 tokenizer,
                 processed_data=None,
                 **kwargs):
        raw_data = json.load(open(data_path, 'r')) if data_path else []
        self.tokenizer = tokenizer
        self.model = model
        self.data = []
        if processed_data:
            self.data = processed_data
        for item in raw_data:
            try:
                query = item['query'].replace('_', ' ')
                context = item['context'].replace('_', ' ')
                label = item['label']
                pair_item = {
                    "texts": [
                        self.tokenizer(query),
                        self.tokenizer(context)
                    ],
                    "label": label
                }
                self.data.append(pair_item)
            except Exception as e:
                print(e.args)
        super().__init__(
            data_path=None,
            processed_data=self.data,
            model=model
        )
        
class PosOnlyPairDataset(TripletDataset):
    def __init__(self, 
                 data_path,
                 model,
                 tokenizer,
                 processed_data=None,
                 **kwargs):
        raw_data = json.load(open(data_path, 'r')) if data_path else []
        self.tokenizer = tokenizer
        self.model = model
        self.data = []
        if processed_data:
            self.data = processed_data
        for item in raw_data:
            try:
                query = item['query'].replace('_', ' ')
                context = item['context'].replace('_', ' ')
                label = item['label']
                pair_item = {
                    "texts": [
                        self.tokenizer(query),
                        self.tokenizer(context)
                    ]
                }
                self.data.append(pair_item)
            except Exception as e:
                print(e.args)
        super().__init__(
            data_path=None,
            processed_data=self.data,
            model=model
        )