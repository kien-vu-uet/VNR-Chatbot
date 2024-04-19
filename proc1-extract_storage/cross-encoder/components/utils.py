import transformers
from torch.utils.data import random_split, DataLoader
import torch
from components import dataset as data_modules
import os
import json

def load_backbone(**kwargs):
    config_abstract = getattr(transformers, f'{kwargs["model"]["model_type"]}Config')
    config = config_abstract(
        vocab_size=kwargs['tokenizer'].vocab_size + 1,
        pad_token_id=kwargs['tokenizer'].pad_token_id,
        bos_token_id=kwargs['tokenizer'].bos_token_id,
        eos_token_id=kwargs['tokenizer'].eos_token_id,
        torch_dtype=torch.float32,
        **kwargs['model']
    )
    # print(config)
    model_abstract = getattr(transformers, f'{kwargs["model"]["model_type"]}ForSequenceClassification')
    model = model_abstract(config)
    if kwargs['pretrained_path'] is not None:
        try:
            if kwargs['load_state_dict_option'] == 'encoder_only':
                encoder = getattr(model, kwargs["model"]["model_type"].lower())
                encoder_weight = getattr(torch.load(kwargs['pretrained_path']), kwargs["model"]["model_type"].lower())
                encoder.load_state_dict(encoder_weight.state_dict())
            else:
                model = torch.load(kwargs['pretrained_path'])
        except Exception as e:
            print('Failed to load encoder checkpoints! Got', e.args)
    return model

def split_train_test(dataset, **kwargs):
    test_size = kwargs['test_size']
    test_length = int(dataset.__len__() * test_size)
    train_length = dataset.__len__() - test_length
    train_set, test_set = random_split(dataset, lengths=[train_length, test_length], \
                                    generator=torch.Generator().manual_seed(kwargs['seed']))
    return train_set, test_set

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['labels'] = torch.stack(labels)

    return batch

def collate_fn_for_classification(batch):
    bsz = len(batch)
    max_len = batch[0]['input_ids'].__len__()
    input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.float32) for item in batch]
    token_type_ids = [torch.tensor(item['token_type_ids'], dtype=torch.long) for item in batch]
    labels = [torch.tensor(item['labels'], dtype=torch.long) for item in batch]

    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['labels'] = torch.stack(labels)
    batch['position_ids'] = torch.arange(0, max_len, dtype=torch.long).expand(bsz, -1)
    return batch

def collate_fn_for_regression(batch):
    bsz = len(batch)
    max_len = batch[0]['input_ids'].__len__()
    input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.float32) for item in batch]
    token_type_ids = [torch.tensor(item['token_type_ids'], dtype=torch.long) for item in batch]
    labels = [torch.tensor(item['labels'], dtype=torch.float32) for item in batch]

    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['labels'] = torch.stack(labels)
    batch['position_ids'] = torch.arange(0, max_len, dtype=torch.long).expand(bsz, -1)
    return batch

def get_dataset(**kwargs):
    dataset_abstract = getattr(data_modules, kwargs['data_module'])
    if os.path.exists(kwargs['train_path']) and os.path.exists(kwargs['test_path']):
        return dataset_abstract.load_from_disk(kwargs['train_path']), \
                           dataset_abstract.load_from_disk(kwargs['test_path'])                
    
    train_set, test_set = split_train_test(dataset_abstract(**kwargs), **kwargs)
    print(f"Save training set to {kwargs['train_path']}")
    json.dump(
            {   
                "max_length": kwargs['max_length'],
                "tokenizer": kwargs['tokenizer'].name_or_path,
                "data": [train_set.dataset.__getitem__(idx) for idx in train_set.indices], 
            },              
            open(kwargs['train_path'], 'w')
    )
    print(f"Save testing set to {kwargs['test_path']}")
    json.dump(
            {   
                "max_length": kwargs['max_length'],
                "tokenizer": kwargs['tokenizer'].name_or_path,
                "data": [test_set.dataset.__getitem__(idx) for idx in test_set.indices], 
            },              
            open(kwargs['test_path'], 'w')
    )
    return train_set, test_set
        
    
def get_dataloader(dataset, dtype='train', **kwargs):
    batch_size = kwargs['batch_size']
    shuffle = kwargs[f'shuffle_{dtype}']
    if kwargs['model']['problem_type'] == 'regression':
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_for_regression)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_for_classification)
