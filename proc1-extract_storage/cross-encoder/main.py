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
from lightning.pytorch.trainer import Trainer

start = time.time()

def calculate_time(curr_ep: int, max_epochs) -> str:
    execute_time = time.time()-start
    hh = execute_time // 3600
    mm = (execute_time % 3600) // 60
    ss = (execute_time % 3600) % 60
    return f"Execute time: {int(hh)}h-{int(mm):02d}m-{int(ss):02d}s; Progress: {curr_ep}/{max_epochs};"

def check_batch_shape(batch, opt) -> bool:
    for k, v in batch.items():
        if k != 'labels' and v.size() != torch.Size([opt['batch_size'], opt['max_length']]): 
            return False
        elif k == 'labels' and v.size() != torch.Size([opt['batch_size']]):
            return False
    return True

def train(model, device, data_loader, optimizer, opt):
    if not os.path.exists(opt['model_checkpoints']): 
        os.mkdir(opt['model_checkpoints'])
    EPOCHS = opt['max_epochs']
    scores = []
    model.to(device)
    model.train()
    for epoch in range(opt['start_epoch'], EPOCHS+1):  # loop over the dataset multiple times
        t_loss, tp, tn, fp, fn = 0., 0, 0, 0, 0
        desc = f"Epoch {epoch}/{EPOCHS}"
        pbar = tqdm(data_loader, desc=desc)
        for batch_train in pbar:
            # get the inputs;
            # assert check_batch_shape(batch_train, opt), f'Stop iterations!'
            batch_train = {k:v.to(device) for k,v in batch_train.items()}
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            model.train()
            outputs = model(**batch_train)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"loss": loss.cpu().item()})
            t_loss += loss.cpu().item()
            logits = torch.argmax(outputs.logits, dim=-1).flatten()
            labels = batch_train['labels'].flatten()
            if opt['model']['problem_type'] == 'regression':
                logits = torch.round(outputs.logits.flatten())
                labels = torch.round(labels)
                # logits = torch.round(logits / (logits.abs().max() + 1e-8))
                # labels = torch.round(labels / (logits.abs().max() + 1e-8))
            tp += torch.sum(((logits == 1) & (labels == 1))).cpu().item()
            tn += torch.sum(((logits == 0) & (labels == 0))).cpu().item()
            fp += torch.sum(((logits == 1) & (labels == 0))).cpu().item()
            fn += torch.sum(((logits == 0) & (labels == 1))).cpu().item()
            batch_train = {k:v.cpu() for k,v in batch_train.items()}
        avg_loss = -t_loss / len(data_loader) 
        acc = (tp + tn) / (tp + tn + fp + fn)
        pre = (tp + 1e-8) / (tp + fp + 1e-8)
        rec = (tp + 1e-8) / (tp + fn + 1e-8)
        f1 = (2 * pre * rec) / (pre + rec + 1e-8)
        scores.append([avg_loss, acc, f1])         

        torch.save(model, f"{opt['model_checkpoints']}/epoch_{epoch}.pt")
        print(f"{calculate_time(epoch, EPOCHS)} loss = {-avg_loss:.4f}; acc = {acc:.2f}; f1 = {f1:.2f}")
        
    scores = torch.tensor(scores, dtype=torch.float16, device=device).T
    top_scores = torch.topk(scores, k=1, largest=True, dim=-1)
    print("Best optimized loss model: ", f"{opt['model_checkpoints']}/epoch_{top_scores.indices[0].item() + opt['start_epoch']}.pt", f"Loss: {-top_scores.values[0].item() + opt['start_epoch']:.4f}")
    print("Best accurated model: ", f"{opt['model_checkpoints']}/epoch_{top_scores.indices[1].item() + opt['start_epoch']}.pt", f"Acc: {top_scores.values[1].item() + opt['start_epoch']:.2f}")
    print("Best optimized f1 model: ", f"{opt['model_checkpoints']}/epoch_{top_scores.indices[2].item() + opt['start_epoch']}.pt", f"F1: {top_scores.values[2].item() + opt['start_epoch']:.2f}")


def test(device, data_loader, opt):
    EPOCHS = opt['max_epochs']
    scores = []
    for epoch in range(0, EPOCHS+1):  # loop over the dataset multiple times
        if not os.path.exists(f"{opt['model_checkpoints']}/epoch_{epoch}.pt"): continue
        model = torch.load(f"{opt['model_checkpoints']}/epoch_{epoch}.pt", map_location=device)
        model.eval()
        t_loss, tp, tn, fp, fn = 0., 0, 0, 0, 0
        desc = f"Epoch {epoch}/{EPOCHS}"
        pbar = tqdm(data_loader, desc=desc)
        for batch_test in pbar:
            # get the inputs;
            # assert check_batch_shape(batch_test, opt), f'Stop iterations!'
            batch_test = {k:v.to(device) for k,v in batch_test.items()}
            
            # forward + backward + optimize
            with torch.no_grad():
                outputs = model(**batch_test)
                loss = outputs.loss
                
                pbar.set_postfix({"loss": loss.cpu().item()})
                t_loss += loss.cpu().item()
                logits = torch.argmax(outputs.logits, dim=-1).flatten()
                labels = batch_test['labels'].flatten()
                if opt['model']['problem_type'] == 'regression':
                    logits = torch.round(outputs.logits.flatten())
                    labels = torch.round(labels)
                    # logits = torch.round(logits / (logits.abs().max() + 1e-8))
                    # labels = torch.round(labels / (logits.abs().max() + 1e-8))
                tp += torch.sum(((logits == 1) & (labels == 1))).cpu().item()
                tn += torch.sum(((logits == 0) & (labels == 0))).cpu().item()
                fp += torch.sum(((logits == 1) & (labels == 0))).cpu().item()
                fn += torch.sum(((logits == 0) & (labels == 1))).cpu().item()
            batch_test = {k:v.cpu() for k,v in batch_test.items()}
            
        avg_loss = -t_loss / len(data_loader) 
        acc = (tp + tn) / (tp + tn + fp + fn)
        pre = (tp + 1e-8) / (tp + fp + 1e-8)
        rec = (tp + 1e-8) / (tp + fn + 1e-8)
        f1 = (2 * pre * rec) / (pre + rec + 1e-8)
        scores.append([avg_loss, acc, f1])            
        
        print(f"{calculate_time(epoch, EPOCHS)} loss = {-avg_loss:.4f}; acc = {acc:.2f}; f1 = {f1:.2f}")
        del model

    scores = torch.tensor(scores, dtype=torch.float, device=device).T
    top_scores = torch.topk(scores, k=1, largest=True, dim=-1)
    print("Best optimized loss model: ", f"{opt['model_checkpoints']}/epoch_{top_scores.indices[0].item()}.pt", f"Loss: {-top_scores.values[0].item():.4f}")
    print("Best accurated model: ", f"{opt['model_checkpoints']}/epoch_{top_scores.indices[1].item()}.pt", f"Acc: {top_scores.values[1].item():.2f}")
    print("Best optimized f1 model: ", f"{opt['model_checkpoints']}/epoch_{top_scores.indices[2].item()}.pt", f"F1: {top_scores.values[2].item():.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file", type=str, default='./config/roberta-base.yml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        opt = yaml.safe_load(f)
        print(yaml.dump(opt, default_flow_style=False, indent=4, explicit_start=True, explicit_end=True, sort_keys=False))
        f.close()
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    opt['tokenizer'] = AutoTokenizer.from_pretrained(opt['tokenizer'], cache_dir=opt['hf_cache'])
    total_train_set, total_test_set = [], []
    for k, v in opt['datasets'].items():
        print(f'Make {k} dataset!')
        train_set, test_set = get_dataset(**v, **opt)
        total_train_set.append(train_set)
        total_test_set.append(test_set)
    total_train_set = ConcatDataset(total_train_set)
    total_test_set = ConcatDataset(total_test_set)
    print('Num of training samples:', len(total_train_set))
    print('Num of testing samples:', len(total_test_set))
    train_loader = get_dataloader(total_train_set, 'train', **opt)
    test_loader = get_dataloader(total_test_set, 'test', **opt)
    
    model = load_backbone(**opt)
    # print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total parameters:', pytorch_total_params)
    opt['start_epoch'] = 1
    if opt['train_from_last_epoch'] and os.path.exists(opt['model_checkpoints']):
        try:
            files = os.listdir(opt['model_checkpoints'])
            files = sorted(files, key= lambda x : int(re.search(r'\d+', x).group(0)), reverse=True)
            last_ckpt = os.path.join(opt['model_checkpoints'], files[0])
            last_ep = int(files[0].lstrip('epoch_').rstrip('.pt'))
            opt['start_epoch'] = last_ep + 1
            model.load_state_dict(torch.load(last_ckpt).state_dict())
            print('Continue training from epoch', opt['start_epoch'])
        except Exception as e:
            print('Error loading last model checkpoint:', e.args)
            print('The training process is still going on!')
    
    optimizer = getattr(torch.optim, opt['optimizer'])(model.parameters(), lr=opt['lr'])
    os.environ['CUDA_LAUNCH_BLOCKING']='1'
    device = torch.device(opt['device'])    
    # torch.cuda.seed_all()
    # if opt['do_train']:
    #     print("TRAINING PROCESS ...")
    #     train(model, device, train_loader, optimizer, opt)
    # print("EVALUATING PROCESS ...")
    # test(device, test_loader, opt)
    

    