import transformers
from torch.utils.data import random_split, DataLoader
import torch
from components import dataset as data_modules
from components import biencoder_dataset as biencoder_data_modules
import os
import json
from collections import OrderedDict

def load_backbone(**kwargs):
    config_abstract = getattr(transformers, f'{kwargs["model"]["model_type"]}Config')
    config = config_abstract(
        vocab_size=kwargs['tokenizer'].vocab_size,
        pad_token_id=kwargs['tokenizer'].pad_token_id,
        # bos_token_id=kwargs['tokenizer'].bos_token_id,
        # eos_token_id=kwargs['tokenizer'].eos_token_id,
        # torch_dtype=torch.float32,
        **kwargs['model']
    )
    # print(config)
    model_abstract = getattr(transformers, f'{kwargs["model"]["model_type"]}ForSequenceClassification')
    model = model_abstract(config)
    if kwargs['pretrained_path'] is not None:
        try:
            if kwargs['load_state_dict_option'] == 'all_matched_keys':
                weights = OrderedDict()
                ckpt_state_dict = torch.load(kwargs['pretrained_path']).state_dict()
                for k,v in model.state_dict().items():
                    if k in ckpt_state_dict.keys() and v.shape == ckpt_state_dict[k].shape:
                        weights[k] = ckpt_state_dict[k]
                    else:
                        weights[k] = v
                model.load_state_dict(weights)
                print('Load all matched keys from checkpoint!')
            elif kwargs['load_state_dict_option'] == 'force_load': 
                print('Force load model object from file!')
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

def collate_fn_for_classification(batch):
    # create new batch
    new_batch = {}
    input_fields = list(batch[0].keys())
    if 'input_ids' in input_fields:
        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
        new_batch['input_ids'] = torch.stack(input_ids)
    if 'attention_mask' in input_fields:
        attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.float32) for item in batch]
        new_batch['attention_mask'] = torch.stack(attention_mask)
    if 'token_type_ids' in input_fields:
        token_type_ids = [torch.tensor(item['token_type_ids'], dtype=torch.long) for item in batch]
        new_batch['token_type_ids'] = torch.stack(token_type_ids)
    if 'position_ids' in input_fields:
        position_ids = [torch.tensor(item['position_ids'], dtype=torch.long) for item in batch]
        new_batch['position_ids'] = torch.stack(position_ids)
    if 'labels' in input_fields:
        labels = [torch.tensor(item['labels'], dtype=torch.long) for item in batch]
        new_batch['labels'] = torch.stack(labels)
    return new_batch

def collate_fn_for_regression(batch):
    # create new batch
    new_batch = {}
    input_fields = list(batch[0].keys())
    if 'input_ids' in input_fields:
        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
        new_batch['input_ids'] = torch.stack(input_ids)
    if 'attention_mask' in input_fields:
        attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.float32) for item in batch]
        new_batch['attention_mask'] = torch.stack(attention_mask)
    if 'token_type_ids' in input_fields:
        token_type_ids = [torch.tensor(item['token_type_ids'], dtype=torch.long) for item in batch]
        new_batch['token_type_ids'] = torch.stack(token_type_ids)
    if 'position_ids' in input_fields:
        position_ids = [torch.tensor(item['position_ids'], dtype=torch.long) for item in batch]
        new_batch['position_ids'] = torch.stack(position_ids)
    if 'labels' in input_fields:
        labels = [torch.tensor(item['labels'], dtype=torch.float) for item in batch]
        new_batch['labels'] = torch.stack(labels)
    return new_batch

def get_dataset(**kwargs):
    dataset_abstract = getattr(data_modules, kwargs['data_module'])
    if os.path.exists(kwargs['train_path']) and os.path.exists(kwargs['test_path']) \
                and not kwargs['force_remake']:
        try:
            return dataset_abstract.load_from_disk(json_path=kwargs['train_path'], **kwargs), \
                           dataset_abstract.load_from_disk(json_path=kwargs['test_path'], **kwargs)    
        except Exception as e:
            print('Cannot load dataset on disk!', e.args)            
    
    train_set, test_set = split_train_test(dataset_abstract(**kwargs), **kwargs)
    if not os.path.exists(os.path.dirname(kwargs['train_path'])): 
        os.mkdir(os.path.dirname(kwargs['train_path']))
    print(f"Save training set to {kwargs['train_path']}")
    train_set.dataset.save_to_disk(kwargs['train_path'], indices=train_set.indices)
    if not os.path.exists(os.path.dirname(kwargs['test_path'])): 
        os.mkdir(os.path.dirname(kwargs['test_path']))
    print(f"Save testing set to {kwargs['test_path']}")
    test_set.dataset.save_to_disk(kwargs['test_path'], indices=test_set.indices)
    return train_set, test_set
        
    
def get_dataloader(dataset, dtype='train', **kwargs):
    batch_size = kwargs['batch_size']
    shuffle = kwargs[f'shuffle_{dtype}']
    num_workers = kwargs['num_workers']
    if kwargs['model']['problem_type'] == 'regression' or \
        kwargs['model']['num_labels'] == 1 or \
            kwargs['model']['problem_type']=='sequence_to_sequence':
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_for_regression, num_workers=num_workers)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_for_classification, num_workers=num_workers)

def get_biencoder_dataset(**kwargs):
    dataset_abstract = getattr(biencoder_data_modules, kwargs['data_module'])
    if os.path.exists(kwargs['train_path']) and os.path.exists(kwargs['test_path']) \
                and not kwargs['force_remake']:
        try:
            return dataset_abstract.load_from_disk(json_path=kwargs['train_path'], **kwargs), \
                           dataset_abstract.load_from_disk(json_path=kwargs['test_path'], **kwargs)    
        except Exception as e:
            print('Cannot load dataset on disk!', e.args)    
    train_set, test_set = split_train_test(dataset_abstract(**kwargs), **kwargs)
    if not os.path.exists(os.path.dirname(kwargs['train_path'])): 
        os.mkdir(os.path.dirname(kwargs['train_path']))
    print(f"Save training set to {kwargs['train_path']}")
    train_set.dataset.save_to_disk(kwargs['train_path'], indices=train_set.indices)
    if not os.path.exists(os.path.dirname(kwargs['test_path'])): 
        os.mkdir(os.path.dirname(kwargs['test_path']))
    print(f"Save testing set to {kwargs['test_path']}")
    test_set.dataset.save_to_disk(kwargs['test_path'], indices=test_set.indices)
    return train_set, test_set
            
def get_biencoder_dataloader(dataset, dtype='train', **kwargs):
    batch_size = kwargs['batch_size']
    shuffle = kwargs[f'shuffle_{dtype}']
    num_workers = kwargs['num_workers']
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)