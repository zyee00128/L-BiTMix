# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:25:30 2023

@author: COCHE User
"""
import os
import gc
import numpy as np
import random
import time
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from datacollection import ECGdataset_prepare_finetuning_sepe
from model_src_ecg.model_code_default import NN_default, NN_default_series, NN_default_replace, NN_default_parallel, Cutmix,Cutmix_student
from model_src_ecg.HM_BiTCN_model_only import HM_BiTCN_Model
from model_src_ecg.ECG_TCN import *
from pytorchtools import EarlyStopping
from evaluation import print_result, find_thresholds
from Half_Trainer import HalfTrainer
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.deterministic = True
    # allow benchmark when shapes are stable for perf
    torch.backends.cudnn.benchmark = True

def mask_ecg_signal(signal, valid_lead_num):
    if valid_lead_num == 1:
        # mask_lead = np.arange(1,12)
        mask_lead = torch.arange(1,12,device=signal.device)
        signal[:, mask_lead, :, :] = 0
        return signal
    elif valid_lead_num == 3:
        # mask_lead = [0, 2, 3, 4, 5, 7, 8, 9, 11] #[1,6,10],II, V1, V5
        mask_lead = torch.tensor([0, 2, 3, 4, 5, 7, 8, 9, 11], device=signal.device)
        signal[:, mask_lead, :, :] = 0
        return signal
    else:
        return signal

def validate(model, valloader, device, iftest=False, threshold=0.5 * np.ones(5), iftrain=False, args=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    probs_list = []
    labels_list = []

    # losses, probs, lbls, logit = [], [], [], []
    # for step, (inp_windows_t, lbl_t) in enumerate(valloader):
    #     inp_windows_t, lbl_t = inp_windows_t.float().to(device), lbl_t.int().to(device)
    #     with torch.no_grad():
    #         inp_windows_t = mask_ecg_signal(inp_windows_t, args.leads)
    #         main_outputs = model(inp_windows_t)
    #         loss = F.binary_cross_entropy_with_logits(main_outputs, lbl_t.float())
            
    #         prob = main_outputs.sigmoid().data.cpu().numpy()
    #         losses.append(loss.item())
    #         probs.append(prob)
    #         lbls.append(lbl_t.data.cpu().numpy())
    #         logit.append(main_outputs.data.cpu().numpy())
    # lbls = np.concatenate(lbls)
    # probs = np.concatenate(probs)

    with torch.no_grad():
        for inp_windows_t, lbl_t in valloader:
            inp_windows_t = inp_windows_t.float().to(device)
            lbl_t = lbl_t.float().to(device)
            inp_windows_t = mask_ecg_signal(inp_windows_t, args.leads)
            # forward
            outputs = model(inp_windows_t)
            loss = F.binary_cross_entropy_with_logits(outputs, lbl_t, reduction='sum')
            total_loss += loss.item()
            total_samples += inp_windows_t.size(0)
            probs_list.append(outputs.sigmoid().cpu())
            labels_list.append(lbl_t.cpu())

    # concatenate on CPU
    probs = torch.cat(probs_list, dim=0).numpy()
    lbls = torch.cat(labels_list, dim=0).numpy()
    mean_loss = total_loss / max(1, total_samples)
    # if iftest:
    #     valid_result = print_result(np.mean(losses), lbls.copy(), probs.copy(), 'test', threshold)
    # elif iftrain:
    #     threshold = find_thresholds(lbls.copy(), probs.copy())
    #     valid_result = print_result(np.mean(losses), lbls.copy(), probs.copy(), 'train', threshold)
    # else:
    #     threshold = find_thresholds(lbls, probs)
    #     valid_result = print_result(np.mean(losses), lbls, probs, 'valid', threshold)
    # neg_ratio = (len(probs) - np.sum(probs, axis=0)) / np.sum(probs, axis=0)
    # valid_result.update({'neg_ratio': neg_ratio})
    # valid_result.update({'threshold': threshold})

    if iftest:
        valid_result = print_result(mean_loss, lbls.copy(), probs.copy(), 'test', threshold)
    elif iftrain:
        found_threshold = find_thresholds(lbls.copy(), probs.copy())
        valid_result = print_result(mean_loss, lbls.copy(), probs.copy(), 'train', found_threshold)
    else:
        found_threshold = find_thresholds(lbls, probs)
        valid_result = print_result(mean_loss, lbls, probs, 'valid', found_threshold)

    neg_ratio = (len(probs) - np.sum(probs, axis=0)) / np.sum(probs, axis=0)
    valid_result.update({'neg_ratio': neg_ratio})
    valid_result.update({'threshold': valid_result.get('threshold', None)})

    return valid_result

def count_parameters(model):
    # for n,p in model.named_parameters():
    #     if p.requires_grad:
    #         print(n)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_pretrained_model(net, path, args, device='cuda:0'):
    pretrained_dict = torch.load(path, map_location=device)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k.find('classifier.1') < 0}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    return net

def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

## HM_BiTCN in series parallel replace only_BiTCN ablution
def MixTcn_Lite(args): 
    # print('ranklist:',args.ranklist)
    # print('semiconfig:', args.semi_config)
    device = args.device
    device_train = args.device_train
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    batch_size = args.batch_size
    # print('batch_size:', batch_size)
    # print('interval',args.interval)
    setup_seed(args.seed)

    all_results = []   

    load_to_device = getattr(args, 'load_data_to_gpu', False)
    fold_datasets = ECGdataset_prepare_finetuning_sepe(args=args, device=device_train, load_to_device=load_to_device)
    for fold_idx, (train_ds, valid_ds, test_ds) in enumerate(fold_datasets):
        release_gpu_memory()
        args.current_fold = fold_idx
        print(f"Fold {fold_idx} 数据集:")
        print(f"  训练集大小: {len(train_ds)}")
        print(f"  验证集大小: {len(valid_ds)}")
        print(f"  测试集大小: {len(test_ds)}")    

        # If data already loaded to device, DataLoader must use num_workers=0 and pin_memory=False
        dl_num_workers = 0 if load_to_device else getattr(args, 'num_workers', 0)
        dl_pin_memory = False if load_to_device else True
        loader_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=dl_pin_memory, num_workers=dl_num_workers)
        loader_valid = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, pin_memory=dl_pin_memory, num_workers=dl_num_workers)
        loader_test = DataLoader(test_ds, batch_size=batch_size, shuffle=True, pin_memory=dl_pin_memory, num_workers=dl_num_workers)

        label_iter = iter(loader_train)
        iteration = len(loader_train) * args.finetune_epoch
        if args.finetune_dataset in ('WFDB_Ga', 'WFDB_ChapmanShaoxing'):
            iteration = iteration * 2
        # print('max_iteration:', iteration)

        path = args.root + '/checkpoint/'
        os.makedirs(path, exist_ok=True)
        model_config = args.model_config
        base_save_name = 'ECG' + args.finetune_dataset + args.semi_config + model_config + args.ranklist + 'ratio' + str(
            args.finetune_label_ratio) + 'seed' + str(args.seed) + str(args.op_config)
        fold_save_name = base_save_name + f'_fold{fold_idx}'
        print(' fold_save_name:', fold_save_name)

        start_time = time.time()
        num_layers, complexity = 14, 64
        if args.op_config == 'HM_BiTCN_series':
            net = NN_default_series(nOUT=args.num_class, complexity=complexity, inputchannel=12, num_layers=num_layers, rank_list=args.r, mix_weight=args.mix_weight)
        elif args.op_config == 'HM_BiTCN_parallel' or args.op_config == 'HM_BiTCN_search':
            net = NN_default_parallel(nOUT=args.num_class, complexity=complexity, inputchannel=12, num_layers=num_layers, rank_list=args.r, mix_weight=args.mix_weight)     
        elif args.op_config == 'HM_BiTCN_replace':
            net = NN_default_replace(nOUT=args.num_class, complexity=complexity, inputchannel=12, num_layers=num_layers, rank_list=args.r, mix_weight=args.mix_weight)  
        elif args.op_config == 'only_HM_BiTCN':
            net = HM_BiTCN_Model(input_channels=12, complexity=complexity, num_classes=args.num_class, dilations=[8, 4, 2, 1], mix_weight=0.3, dropout=0.1)
        elif args.op_config == 'HM_BiTCN_ablution':
            net = NN_default(nOUT=args.num_class, complexity=complexity, inputchannel=12, num_layers=num_layers, rank_list=args.r, mix_weight=args.mix_weight)  

        if args.load_pretrain:
            file_name_pretrain = args.pretrain_dataset + 'tinylight' + 'bias_full_checkpoint.pkl'
            print('loading pretrain')
            net = load_pretrained_model(net, path + file_name_pretrain, args, device=device)
        
        optimizer = optim.AdamW(net.parameters(), lr=0.001)
        net.to(device)
        early_stopping = EarlyStopping(15, verbose=True,dataset_name=fold_save_name,delta=0, args=args)  # 15
        
        # step = 0
        # net.train()
        # setup_seed(args.seed)
        my_lr_scheduler = get_linear_schedule_with_warmup(optimizer,int(iteration*0.01), iteration, last_epoch=-1)
        scaler = GradScaler(device=args.device, enabled=args.enable_amp)
        best_map = -1.0
        best_ckpt = os.path.join(path, fold_save_name + '_best_checkpoint.pkl')
        
        running_loss = 0.0
        base_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        for current in range(iteration):
            if current % args.interval == 0 and current > 0:
                print('training_loss:', running_loss / args.interval)
                running_loss = 0.0
                valid_result = validate(net, loader_valid, device, args=args)
                early_stopping(1 / valid_result['Map_value'], net)
                if valid_result.get('Map_value', 0.0) > best_map:
                    best_map = valid_result['Map_value']
                    torch.save(net.state_dict(), best_ckpt)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            ## mini-batch sampling
            try:
                images, labels = next(label_iter)
            except Exception:
                label_iter = iter(loader_train)
                images, labels = next(label_iter)
            images = images.float().to(device, non_blocking=False)
            labels = labels.float().to(device, non_blocking=False)
            with torch.no_grad():
                images, labels = Cutmix(images, labels, device)
            
            # optimizer.zero_grad()
            # inputs = images
            # outputs = net(inputs)
            # loss = F.binary_cross_entropy_with_logits(outputs, labels)
            # loss.backward()
            # optimizer.step()
            # running_loss += loss.item()
            # step += 1
            # my_lr_scheduler.step()

            with autocast(device_type=args.device.split(":")[0], enabled=args.enable_amp):
                outputs = net(images)
                loss = F.binary_cross_entropy_with_logits(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            my_lr_scheduler.step()
            running_loss += loss.item()

        end_time = time.time()
        peak_absolute = torch.cuda.max_memory_allocated(device)
        train_usage = peak_absolute - base_memory
        print(f"GPU Memory Allocated: {train_usage / 1024 / 1024:.2f} MB")
        running_time = (end_time - start_time) / (current + 1)
        print(f"running time {running_time:.2f} 秒")
        # allocated_memory = torch.cuda.max_memory_allocated(device)
        # print(f"GPU Memory Allocated: {allocated_memory / 1024 / 1024:.2f} MB")
        trainable_num = count_parameters(net)
        print('trainable_num:', trainable_num)
        
        if os.path.exists(best_ckpt):
            try:
                net.load_state_dict(torch.load(best_ckpt, map_location=device))
            except Exception as e:
                print('Failed to load best ckpt:', e)
        
        net.eval()
        with torch.no_grad():
            valid_result = validate(net, loader_valid, device, args=args)
            test_result = validate(net, loader_test, device, iftest=True, 
                                        threshold=valid_result['threshold'],
                                        args=args)
            test_result.update({'trainable_num': trainable_num})
            test_result.update({'time': running_time})
            test_result.update({'memory': train_usage})
            all_results.append(test_result)
        
        del net, optimizer, loader_train, loader_valid, loader_test
        release_gpu_memory()

        avg_result = {}
        keys = all_results[0].keys()   
        for key in keys:
            if isinstance(all_results[0][key], (int, float, np.floating)):
                avg_result[key] = np.mean([r[key] for r in all_results])
                avg_result[key + '_std'] = np.std([r[key] for r in all_results])
        avg_result['memory'] = all_results[0]['memory']
        if len(all_results) > 1:
            avg_result['memory_last4_mean'] = np.mean([r['memory'] for r in all_results[1:]])
        avg_result['config_name'] = str(args.op_config) + '_' + str(args.finetune_dataset) + '_' + str(args.mix_weight)
    return avg_result

def ECG_TCN_compa(args):
    
    device = args.device
    device_train = args.device_train
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    batch_size = args.batch_size
    setup_seed(args.seed)

    all_results = []   
    load_to_device = getattr(args, 'load_data_to_gpu', False)
    fold_datasets = ECGdataset_prepare_finetuning_sepe(args=args, device=device_train, load_to_device=load_to_device)
    for fold_idx, (train_ds, valid_ds, test_ds) in enumerate(fold_datasets):
        release_gpu_memory()
        args.current_fold = fold_idx
        print(f"Fold {fold_idx} 数据集:")
        print(f"  训练集大小: {len(train_ds)}")
        print(f"  验证集大小: {len(valid_ds)}")
        print(f"  测试集大小: {len(test_ds)}")    

        # If data already loaded to device, DataLoader must use num_workers=0 and pin_memory=False
        dl_num_workers = 0 if load_to_device else getattr(args, 'num_workers', 0)
        dl_pin_memory = False if load_to_device else True
        loader_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=dl_pin_memory, num_workers=dl_num_workers)
        loader_valid = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, pin_memory=dl_pin_memory, num_workers=dl_num_workers)
        loader_test = DataLoader(test_ds, batch_size=batch_size, shuffle=True, pin_memory=dl_pin_memory, num_workers=dl_num_workers)

        label_iter = iter(loader_train)
        iteration = len(loader_train) * args.finetune_epoch
        if args.finetune_dataset in ('WFDB_Ga', 'WFDB_ChapmanShaoxing'):
            iteration = iteration * 2
        # print('max_iteration:', iteration)

        path = args.root + '/checkpoint/'
        os.makedirs(path, exist_ok=True)
        model_config = args.model_config
        base_save_name = 'ECG' + args.finetune_dataset + args.semi_config + model_config + args.ranklist + 'ratio' + str(
            args.finetune_label_ratio) + 'seed' + str(args.seed) + str(args.op_config)
        fold_save_name = base_save_name + f'_fold{fold_idx}'
        print(' fold_save_name:', fold_save_name)

        start_time = time.time()
        num_layers, complexity = 8, 64
        net = ECG_TCN(input_channels=12, complexity=complexity, num_classes=args.num_class, num_layers=num_layers, dropout=0.1)
        
        optimizer = optim.AdamW(net.parameters(), lr=1e-3)
        net.to(device)
        early_stopping = EarlyStopping(15, verbose=True,dataset_name=fold_save_name,delta=0, args=args)  # 15
        
        my_lr_scheduler = get_linear_schedule_with_warmup(optimizer,int(iteration*0.01), iteration, last_epoch=-1)
        scaler = GradScaler(device=args.device, enabled=False)
        best_map = -1.0
        best_ckpt = os.path.join(path, fold_save_name + '_best_checkpoint.pkl')
        
        running_loss = 0.0
        base_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        for current in range(iteration):
            if current % args.interval == 0 and current > 0:
                print('training_loss:', running_loss / args.interval)
                running_loss = 0.0
                valid_result = validate(net, loader_valid, device, args=args)
                early_stopping(1 / valid_result['Map_value'], net)
                if valid_result.get('Map_value', 0.0) > best_map:
                    best_map = valid_result['Map_value']
                    torch.save(net.state_dict(), best_ckpt)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            ## mini-batch sampling
            try:
                images, labels = next(label_iter)
            except Exception:
                label_iter = iter(loader_train)
                images, labels = next(label_iter)
            images = images.float().to(device, non_blocking=False)
            labels = labels.float().to(device, non_blocking=False)
            with torch.no_grad():
                images, labels = Cutmix(images, labels, device)
        
            with autocast(device_type=args.device.split(":")[0], enabled=False):
                outputs = net(images)
                loss = F.binary_cross_entropy_with_logits(outputs, labels)

            scaler.scale(loss).backward()
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            my_lr_scheduler.step()
            running_loss += loss.item()

        end_time = time.time()
        peak_absolute = torch.cuda.max_memory_allocated(device)
        train_usage = peak_absolute - base_memory
        print(f"GPU Memory Allocated: {train_usage / 1024 / 1024:.2f} MB")
        running_time = (end_time - start_time) / (current + 1)
        print(f"running time {running_time:.2f} 秒")
        # allocated_memory = torch.cuda.max_memory_allocated(device)
        # print(f"GPU Memory Allocated: {allocated_memory / 1024 / 1024:.2f} MB")
        trainable_num = count_parameters(net)
        print('trainable_num:', trainable_num)
        
        if os.path.exists(best_ckpt):
            try:
                net.load_state_dict(torch.load(best_ckpt, map_location=device))
            except Exception as e:
                print('Failed to load best ckpt:', e)
        
        net.eval()
        with torch.no_grad():
            valid_result = validate(net, loader_valid, device, args=args)
            test_result = validate(net, loader_test, device, iftest=True, 
                                        threshold=valid_result['threshold'],
                                        args=args)
            test_result.update({'trainable_num': trainable_num})
            test_result.update({'time': running_time})
            test_result.update({'memory': train_usage})
            all_results.append(test_result)
        
        del net, optimizer, loader_train, loader_valid, loader_test
        release_gpu_memory()

        avg_result = {}
        keys = all_results[0].keys()   
        for key in keys:
            if isinstance(all_results[0][key], (int, float, np.floating)):
                avg_result[key] = np.mean([r[key] for r in all_results])
                avg_result[key + '_std'] = np.std([r[key] for r in all_results])
        avg_result['memory'] = all_results[0]['memory']
        if len(all_results) > 1:
            avg_result['memory_last4_mean'] = np.mean([r['memory'] for r in all_results[1:]])
        avg_result['config_name'] = str(args.op_config) + '_' + str(args.finetune_dataset) + '_' + str(args.mix_weight)
    return avg_result

def release_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()