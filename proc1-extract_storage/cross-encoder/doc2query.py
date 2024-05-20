from components import *
import argparse
import yaml
import torch
import os
import re
import time
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from tqdm import tqdm
# from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.utils.data import ConcatDataset
# from lightning.pytorch.trainer import Trainer
import wandb
import evaluate
import numpy as np

def callback_fn(score=0., epoch=0., steps=0., lr=0., loss=0.):
    wandb.log(
        {
            "evaluate/score": score,
            "train/epoch": epoch,
            "train/steps": steps,
            "train/learning_rate": lr,
            "train/loss": loss
        }
    )

def compute_metrics_fn(eval_preds):
    metric = evaluate.load("bleu")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    predictions = opt['tokenizer'].batch_decode(predictions, skip_special_tokens=True)
    labels = opt['tokenizer'].batch_decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    wandb.login(key='24687e333c06f60cd01a0ff6327c8b872bb4645f', anonymous='never')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file", type=str, default='./config/doc-to-query.yml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        opt = yaml.safe_load(f)
        print(yaml.dump(opt, default_flow_style=False, indent=4, explicit_start=True, explicit_end=True, sort_keys=False))
        f.close()
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    
    model = AutoModel.from_pretrained(opt['pretrained_path'], 
                                cache_dir=opt['hf_cache'],
                                token='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')
    
    os.makedirs(opt['model_checkpoints'], exist_ok=True)

    opt['tokenizer'] = AutoTokenizer.from_pretrained(opt['pretrained_path'])
    total_train_set, total_test_set = [], []
    for k, v in opt['datasets'].items():
        print(f'Make {k} dataset!')
        train_set, test_set = get_biencoder_dataset(**v, **opt)
        total_train_set.append(train_set)
        total_test_set.append(test_set)
    total_train_set = ConcatDataset(total_train_set)
    total_test_set = ConcatDataset(total_test_set)
    print('Num of training samples:', len(total_train_set))
    print('Num of testing samples:', len(total_test_set))
    train_loader = get_biencoder_dataloader(total_train_set, 'train', **opt)
    test_loader = get_biencoder_dataloader(total_test_set, 'test', **opt)
    optimizer = getattr(torch.optim, opt['optimizer'])

    wandb.init(project='Doc-to-Query', name='doc2query-registration-domain-finetuned')

    training_args = TrainingArguments(
        output_dir=opt['model_checkpoints'],
        do_train=True,
        do_eval=True,
        evaluation_strategy='epoch',
        per_device_train_batch_size=opt['batch_size'],
        per_device_eval_batch_size=opt['batch_size'],
        num_train_epochs=opt['max_epochs'],
        warmup_steps=100, 
        optim='adamw',
        learning_rate=opt['lr'],
        adam_beta1=0.9, 
        adam_beta2=0.999,
        # adafactor=True,
        logging_dir='./training_logs',
        save_strategy='epoch',
        save_safetensors=True,
        use_cpu=False,
        label_names=['labels'],
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        push_to_hub=True,
        hub_model_id='kien-vu-uet/doc2query-msmarco-vietnamese-mt5-base-v1-finetuned',
        hub_token='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',
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_loader,
        eval_dataset=test_loader,
        tokenizer=opt['tokenizer'],
        compute_metrics=compute_metrics_fn
    )

    # %%
    trainer.train()
    

    
    