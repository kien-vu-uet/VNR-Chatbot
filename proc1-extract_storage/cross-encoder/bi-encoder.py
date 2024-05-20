from components import *
import argparse
import yaml
import torch
import os
import re
import time
from transformers import AutoTokenizer
from tqdm import tqdm
# from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.utils.data import ConcatDataset
# from lightning.pytorch.trainer import Trainer
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import evaluation
from pyvi.ViTokenizer import tokenize
import wandb

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

if __name__ == "__main__":
    wandb.login(key='24687e333c06f60cd01a0ff6327c8b872bb4645f', anonymous='never')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file", type=str, default='./config/bi-encoder-pair.yml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        opt = yaml.safe_load(f)
        print(yaml.dump(opt, default_flow_style=False, indent=4, explicit_start=True, explicit_end=True, sort_keys=False))
        f.close()
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    
    model = SentenceTransformer(opt['pretrained_path'], 
                                cache_folder=opt['hf_cache'],
                                device=opt['device'],
                                token='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')
    os.makedirs(opt['model_checkpoints'], exist_ok=True)

    opt['tokenizer'] = tokenize
    total_train_set, total_test_set = [], []
    for k, v in opt['datasets'].items():
        print(f'Make {k} dataset!')
        train_set, test_set = get_biencoder_dataset(**dict(**v, **opt, model=model))
        total_train_set.append(train_set)
        total_test_set.append(test_set)
    total_train_set = ConcatDataset(total_train_set)
    total_test_set = ConcatDataset(total_test_set)
    print('Num of training samples:', len(total_train_set))
    print('Num of testing samples:', len(total_test_set))
    train_loader = get_biencoder_dataloader(total_train_set, 'train', **opt)
    test_loader = get_biencoder_dataloader(total_test_set, 'test', **opt)
    criterion = getattr(losses, opt['criterion'])(model)
    optimizer = getattr(torch.optim, opt['optimizer'])
    print(total_test_set.datasets)
    eval_examples = []
    for ds in total_test_set.datasets:
        eval_examples.extend(ds.input_examples)
    evaluator = getattr(evaluation, f"{opt['problem_type']}Evaluator").from_input_examples(
        eval_examples,
        batch_size=opt['batch_size']
    )
    # wandb.init(project='Bi-Encoder', name='SimCSE-finetuned-registration-domain')
    # model.fit(
    #     train_objectives=[(train_loader, criterion)],
    #     evaluator=evaluator,
    #     epochs=opt['max_epochs'],
    #     warmup_steps=100,
    #     optimizer_class=optimizer,
    #     optimizer_params={'lr': opt['lr']},
    #     save_best_model=True,
    #     weight_decay=0.1,
    #     show_progress_bar=True,
    #     log_steps=100,
    #     evaluation_steps=100,
    #     callback=callback_fn,
    #     output_path=opt['model_checkpoints']
    # )
    
    # print(model.push_to_hub(f"kien-vu-uet/{opt['model']['problem_type'].lower()}-finetuned-biencoder",
    #                   token='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',
    #                   local_model_path=opt['model_checkpoints']))
    

    
    