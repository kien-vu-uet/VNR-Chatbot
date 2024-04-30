from torch.utils.data import Dataset
import json, csv
from transformers import AutoTokenizer
import os
os.environ['TRANSFORMERS_CACHE'] = '../../hf_cache/'

class ViNLIZaloDataset(Dataset):
    def __init__(self, 
                 data_path='../data/ViNLI-Zalo-supervised.json', 
                 processed_data=None,
                 tokenizer='vinai/phobert-base-v2',
                 max_length=515,
                 input_fields=['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'position_ids'],
                 reverse_input=True,
                 **kwargs):
        raw_data = json.load(open(data_path, 'r')) if data_path else []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) \
                        if isinstance(tokenizer, str) else tokenizer
        self.max_length = max_length
        self.input_fields = input_fields
        self.reverse_input = reverse_input
        self.data = []
        if processed_data is not None:
            self.data = processed_data
        position_ids = list(range(0, self.max_length, 1))
        for item_ in raw_data:
            try:
                question = item_['anchor']
                context_p = item_['pos']
                context_n = item_['hard_neg']                
                p_item = dict(self.tokenizer(text=question, 
                                                text_pair=context_p, 
                                                max_length=self.max_length, 
                                                padding=False, 
                                                truncation='only_second', 
                                                return_token_type_ids=True,
                                                return_attention_mask=True))
                p_item.update({
                    "position_ids": position_ids,
                    "labels": int(1)
                    })
                self.data.append(p_item)
                
                if self.reverse_input:
                    p_item_reversed = dict(self.tokenizer(text=context_p, 
                                                            text_pair=question, 
                                                            max_length=self.max_length, 
                                                            padding=False, 
                                                            truncation='only_first', 
                                                            return_token_type_ids=True,
                                                            return_attention_mask=True))
                    p_item_reversed.update({
                        "position_ids": position_ids,
                        "labels": int(1)
                        })
                    self.data.append(p_item_reversed)
                
                n_item = dict(self.tokenizer(text=question, 
                                                text_pair=context_n, 
                                                max_length=self.max_length, 
                                                padding=False, 
                                                truncation='only_second', 
                                                return_token_type_ids=True,
                                                return_attention_mask=True))
                n_item.update({
                    "position_ids": position_ids,
                    "labels": int(0)
                    })
                self.data.append(n_item)
                
                if self.reverse_input:
                    n_item_reversed = dict(self.tokenizer(text=context_n, 
                                                            text_pair=question, 
                                                            max_length=self.max_length, 
                                                            padding=False, 
                                                            truncation='only_first', 
                                                            return_token_type_ids=True,
                                                            return_attention_mask=True))
                    n_item_reversed.update({
                        "position_ids": position_ids,
                        "labels": int(0)
                        })
                    self.data.append(n_item_reversed)
            except Exception as e:
                print(e.args)
            
    def __len__(self):
        return len(self.data)
    
    def padding(self, item):
        if item['input_ids'].__len__() < self.max_length:
            pad_len = self.max_length - item['input_ids'].__len__() 
            item['input_ids'] += [self.tokenizer.pad_token_id] * pad_len
            item['token_type_ids'] += [1] * pad_len
            item['attention_mask'] += [0] * pad_len
            item['position_ids'] += list(range(len(item['position_ids']), self.max_length))
        elif item['input_ids'].__len__() > self.max_length:
            item['input_ids'] = item['input_ids'][:self.max_length]
            item['token_type_ids'] = item['token_type_ids'][:self.max_length]
            item['attention_mask'] = item['attention_mask'][:self.max_length]
            item['position_ids'] = item['position_ids'][:self.max_length]
        return item
    
    def __getitem__(self, index):
        data_item = self.padding(self.data[index])
        return {k:v for k,v in data_item.items() if k in self.input_fields}
    
    def save_to_disk(self, json_path, indices=None):
        saved_data = self.data if indices is None else [self.data[idx] for idx in indices]
        if not os.path.exists(os.path.dirname(json_path)): 
            os.mkdir(os.path.dirname(json_path))
        json.dump(
            {   
                "max_length": self.max_length,
                "tokenizer": self.tokenizer.name_or_path,
                "data": saved_data, 
                "input_fields": self.input_fields,
                "reverse_input": self.reverse_input
            },              
            open(json_path, 'w')
        )
        
    @classmethod
    def load_from_disk(cls, json_path, **kwargs):
        params = json.load(open(json_path, 'r'))
        tokenizer = params['tokenizer']
        assert tokenizer == kwargs['tokenizer'] if isinstance(kwargs['tokenizer'], str) \
                            else kwargs['tokenizer'].name_or_path, f'Tokenizer card does not match!'
        max_length = kwargs['max_length']
        input_fields = kwargs['input_fields']
        reverse_input = params['reverse_input']
        data = params['data']
        return cls(data_path=None, 
                    processed_data=data,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    input_fields=input_fields,
                    reverse_input=reverse_input)
        
class ViNLIZaloRegressionDataset(ViNLIZaloDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['labels'] = float(item['labels']) * 100
        return item
    
    
class ViSTSRegressionDataset(ViNLIZaloDataset):
    def __init__(self, 
                data_path='../data/ViNLI-Zalo-supervised.json', 
                processed_data=None,
                tokenizer='vinai/phobert-base-v2',
                max_length=515,
                input_fields=['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'position_ids'],
                reverse_input=True,
                **kwargs):
        raw_data = json.load(open(data_path, 'r')) if data_path else []
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.input_fields = input_fields
        self.reverse_input = reverse_input
        self.data = []
        if processed_data: 
            self.data = processed_data
        position_ids = list(range(0, self.max_length))
        for item_ in raw_data:
            # try:
                sent1 = item_['sentence1']
                sent2 = item_['sentence2']
                score = item_['score']
                new_item = dict(self.tokenizer(text=sent1, 
                                                text_pair=sent2, 
                                                max_length=self.max_length, 
                                                padding=False, 
                                                truncation='only_second',
                                                return_token_type_ids=True,
                                                return_attention_mask=True))
                new_item.update({
                    "labels": round(score / 5, 2),
                    "position_ids": position_ids,
                })
                self.data.append(new_item)
                
                if self.reverse_input:
                    new_item_reversed = dict(self.tokenizer(text=sent2, 
                                                            text_pair=sent1, 
                                                            max_length=self.max_length, 
                                                            padding=False, 
                                                            truncation='only_second',
                                                            return_token_type_ids=True,
                                                            return_attention_mask=True))
                    new_item_reversed.update({
                        "labels": round(score / 5, 2),
                        "position_ids": position_ids,
                    })
                    self.data.append(new_item_reversed)
                
            # except Exception as e:
            #     print(e.args)
                
        super().__init__(None, self.data, self.tokenizer, self.max_length, self.input_fields, self.reverse_input, **kwargs)
        
        
class ViSTSDataset(ViSTSRegressionDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['labels'] = round(item['labels'])
        return item
    
            
        
class IRSegmentDataset(ViNLIZaloDataset):
    def __init__(self, 
                 data_path='../data/train_IR_segment.json', 
                 processed_data=None, 
                 tokenizer='vinai/phobert-base-v2', 
                 max_length=515, 
                 input_fields=['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'position_ids'],
                 reverse_input=True,
                 **kwargs):
        raw_data = json.load(open(data_path, 'r')) if data_path else []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) \
                        if isinstance(tokenizer, str) else tokenizer
        self.max_length = max_length
        self.data = []
        self.input_fields = input_fields
        self.reverse_input = reverse_input
        if processed_data:
            self.data = processed_data
        position_ids = list(range(0, self.max_length, 1))
        for item in raw_data:
            try:
                question = item['question']
                context = item['context']
                label = int(item['labels'])
                p_item = dict(self.tokenizer(text=question, 
                                            text_pair=context, 
                                            max_length=514, 
                                            padding=False, 
                                            truncation='only_second', 
                                            return_token_type_ids=True,
                                            return_attention_mask=True))
                p_item.update({
                    'labels': label,
                    'position_ids': position_ids
                })
                self.data.append(p_item)
                
                if self.reverse_input:
                    p_item_reversed = dict(self.tokenizer(text=context, 
                                                        text_pair=question, 
                                                        max_length=514, 
                                                        padding=False, 
                                                        truncation='only_first', 
                                                        return_token_type_ids=True,
                                                        return_attention_mask=True))
                    p_item_reversed.update({
                        'labels': label,
                        'position_ids': position_ids
                    })
                    self.data.append(p_item_reversed)
            except Exception as e:
                print(e.args)
        super().__init__(None, self.data, self.tokenizer, self.max_length, self.input_fields, self.reverse_input, **kwargs)
        
        
class IRSegmentRegressionDataset(IRSegmentDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['labels'] = float(item['labels']) * 100
        return item
    


class ViMMRCSegmentDataset(IRSegmentDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
class ViMMRCSegmentRegressionDataset(IRSegmentDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['labels'] = float(item['labels']) * 100
        return item
    
    
class BasicNLIDataset(ViNLIZaloDataset):
    def __init__(self, 
                 data_path='/workspace/nlplab/nmq/KLTN/training_model/data/tdt_data/nli/data_nli_0_1_segment.csv', 
                 processed_data=None, 
                 tokenizer='vinai/phobert-base-v2', 
                 max_length=515, 
                 input_fields=['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'position_ids'],
                 reverse_input=True,
                 **kwargs):
        raw_data = []
        if data_path:
            with open(data_path, 'r') as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    raw_data.append(row)
                f.close()
            
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) \
                        if isinstance(tokenizer, str) else tokenizer
        self.max_length = max_length
        self.data = []
        self.input_fields = input_fields
        self.reverse_input = reverse_input
        if processed_data:
            self.data = processed_data
        position_ids = list(range(0, self.max_length, 1))
        for item in raw_data:
            try:
                question = item['query']
                context = item['context']
                label = int(item['label'])
                p_item = dict(self.tokenizer(text=question, 
                                            text_pair=context, 
                                            max_length=514, 
                                            padding=False, 
                                            truncation='only_second', 
                                            return_token_type_ids=True,
                                            return_attention_mask=True))
                p_item.update({
                    'labels': label,
                    'position_ids': position_ids
                })
                self.data.append(p_item)
                
                if self.reverse_input:
                    p_item_reversed = dict(self.tokenizer(text=context, 
                                                        text_pair=question, 
                                                        max_length=514, 
                                                        padding=False, 
                                                        truncation='only_first', 
                                                        return_token_type_ids=True,
                                                        return_attention_mask=True))
                    p_item_reversed.update({
                        'labels': label,
                        'position_ids': position_ids
                    })
                    self.data.append(p_item_reversed)
            except Exception as e:
                print(e.args)
        super().__init__(None, self.data, self.tokenizer, self.max_length, self.input_fields, self.reverse_input, **kwargs)