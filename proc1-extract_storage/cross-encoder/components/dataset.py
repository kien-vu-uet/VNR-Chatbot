from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer
import os
os.environ['TRANSFORMERS_CACHE'] = '../../hf_cache/'

class ViNLIZaloDataset(Dataset):
    def __init__(self, 
                 data_path='../data/ViNLI-Zalo-supervised.json', 
                 processed_data=None,
                 tokenizer='vinai/phobert-base-v2',
                 max_length=515,
                 **kwargs):
        raw_data = open(data_path, 'r').readlines() if data_path else []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) \
                        if isinstance(tokenizer, str) else tokenizer
        self.max_length = max_length
        self.data = []
        if processed_data is not None:
            self.data = processed_data
        for item in raw_data:
            try:
                item_ = json.loads(item)
                question = item_['anchor']
                context_p = item_['pos']
                context_n = item_['hard_neg']
                q_tokens = self.tokenizer(question, padding=False)
                c_len = self.max_length - q_tokens.input_ids.__len__() + 1
                assert c_len >= self.max_length // 2
                cp_tokens = self.tokenizer(context_p, max_length=c_len, padding=False, truncation=True)
                cn_tokens = self.tokenizer(context_n, max_length=c_len, padding=False, truncation=True)
                p_item = {
                    "input_ids": q_tokens.input_ids + cp_tokens.input_ids[1:],
                    "token_type_ids": q_tokens.token_type_ids + [1] * cp_tokens.token_type_ids[1:].__len__(),
                    "attention_mask": q_tokens.attention_mask + cp_tokens.attention_mask[1:],
                    "labels": int(1)
                }
                self.data.append(p_item)
                p_item_reversed = {
                    "input_ids": cp_tokens.input_ids + q_tokens.input_ids[1:],
                    "token_type_ids": cp_tokens.token_type_ids + [1] * q_tokens.token_type_ids[1:].__len__(),
                    "attention_mask": cp_tokens.attention_mask + q_tokens.attention_mask[1:],
                    "labels": int(1)
                }
                self.data.append(p_item_reversed)
                
                n_item = {
                    "input_ids": q_tokens.input_ids + cn_tokens.input_ids[1:],
                    "token_type_ids": q_tokens.token_type_ids + [1] * cn_tokens.token_type_ids[1:].__len__(),
                    "attention_mask": q_tokens.attention_mask + cn_tokens.attention_mask[1:],
                    "labels": int(0)
                }
                self.data.append(n_item)
                n_item_reversed = {
                    "input_ids": cn_tokens.input_ids + q_tokens.input_ids[1:],
                    "token_type_ids": cn_tokens.token_type_ids + [1] * q_tokens.token_type_ids[1:].__len__(),
                    "attention_mask": cn_tokens.attention_mask + q_tokens.attention_mask[1:],
                    "labels": int(0)
                }
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
        elif item['input_ids'].__len__() > self.max_length:
            item['input_ids'] = item['input_ids'][:self.max_length]
            item['token_type_ids'] = item['token_type_ids'][:self.max_length]
            item['attention_mask'] = item['attention_mask'][:self.max_length]
        return item
    
    def __getitem__(self, index):
        return self.padding(self.data[index])
    
    def save_to_disk(self, json_path):
        json.dump(
            {   
                "max_length": self.max_length,
                "tokenizer": self.tokenizer.name_or_path,
                "data": self.data, 
            },              
            open(json_path, 'w')
        )
        
    @classmethod
    def load_from_disk(cls, json_path):
        params = json.load(open(json_path, 'r'))
        max_length = params['max_length']
        tokenizer = params['tokenizer']
        data = params['data']
        return cls(data_path=None, 
                    processed_data=data,
                    tokenizer=tokenizer,
                    max_length=max_length)
        
class ViNLIZaloRegressionDataset(ViNLIZaloDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['labels'] = float(item['labels'])
        return item
    
    
class ViSTSRegressionDataset(ViNLIZaloDataset):
    def __init__(self, 
                data_path='../data/ViNLI-Zalo-supervised.json', 
                processed_data=None,
                tokenizer='vinai/phobert-base-v2',
                max_length=515,
                **kwargs):
        raw_data = open(data_path, 'r').readlines() if data_path else []
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.data = []
        if processed_data: 
            self.data = processed_data
        for item in raw_data:
            try:
                item_ = json.loads(item)
                sent1 = item_['sentence1']
                sent2 = item_['sentence2']
                score = item_['score']
                tokens = tokenizer([sent1, sent2], max_length=self.max_length // 2, padding=False, truncation=True)
                new_item = {
                    "input_ids": tokens.input_ids[0] + tokens.input_ids[1][1:],
                    "token_type_ids": tokens.token_type_ids[0] + [1] * tokens.token_type_ids[1][1:].__len__(),
                    "attention_mask": tokens.attention_mask[0] + tokens.attention_mask[1][1:],
                    "labels": round(score / 5, 2),
                }
                self.data.append(new_item)
                
                new_item_reversed = {
                    "input_ids": tokens.input_ids[1] + tokens.input_ids[0][1:],
                    "token_type_ids": tokens.token_type_ids[1] + [1] * tokens.token_type_ids[0][1:].__len__(),
                    "attention_mask": tokens.attention_mask[1] + tokens.attention_mask[0][1:],
                    "labels": round(score / 5, 2),
                }
                self.data.append(new_item_reversed)
                
            except Exception as e:
                print(e.args)
                
        super().__init__(None, self.data, self.tokenizer, self.max_length, **kwargs)
        
        
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
                 max_length=515, **kwargs):
        raw_data = json.load(open(data_path, 'r')) if data_path else []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) \
                        if isinstance(tokenizer, str) else tokenizer
        self.max_length = max_length
        self.data = []
        if processed_data:
            self.data = processed_data
        for item in raw_data:
            try:
                question = item['question']
                context = item['context']
                q_tokens = self.tokenizer(question, padding=False)
                c_len = self.max_length - q_tokens.input_ids.__len__() + 1
                assert c_len >= self.max_length // 2
                c_tokens = self.tokenizer(context, max_length=c_len, padding=False, truncation=True)
                p_item = {
                    "input_ids": q_tokens.input_ids + c_tokens.input_ids[1:],
                    "token_type_ids": q_tokens.token_type_ids + [1] * c_tokens.token_type_ids[1:].__len__(),
                    "attention_mask": q_tokens.attention_mask + c_tokens.attention_mask[1:],
                    "labels": int(item['labels'])
                }
                self.data.append(p_item)
                p_item_reversed = {
                    "input_ids": c_tokens.input_ids + q_tokens.input_ids[1:],
                    "token_type_ids": c_tokens.token_type_ids + [1] * q_tokens.token_type_ids[1:].__len__(),
                    "attention_mask": c_tokens.attention_mask + q_tokens.attention_mask[1:],
                    "labels": int(item['labels'])
                }
                self.data.append(p_item_reversed)
            except Exception as e:
                print(e.args)
        super().__init__(None, self.data, self.tokenizer, self.max_length, **kwargs)
        
        
class IRSegmentRegressionDataset(IRSegmentDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['labels'] = float(item['labels'])
        return item
    


class ViMMRCSegmentDataset(IRSegmentDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
class ViMMRCSegmentRegressionDataset(IRSegmentDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['labels'] = float(item['labels'])
        return item