# -*- coding: utf-8 -*-
"""
Modified datacollection.py for L-BiTMix integration.
"""
import os
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.model_selection import KFold
import random
import h5py
import csv
import argparse

from helper_code import *
from preprocess import *

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def ECGdataset_prepare_finetuning_sepe(args, device=None, load_to_device=False):
    device = args.device_train
    dataset_dir = os.path.join(args.root, 'Preprocessed_dataset')
    os.chdir(dataset_dir)
    
    file_name = 'class_sepe'+str(args.num_class)+'_dataset_' + args.finetune_dataset + '_'+'32.hdf5'
    print(f" 正在加载缓存数据集: {file_name}")
    hf = h5py.File(file_name, 'r')
    
    train_label_set = np.array(hf.get('label_set'))
    train_record_set = np.array(hf.get('record_set'))
    hf.close()
    
    print(f" 共加载 {len(train_label_set)} 个样本")
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_datasets = []

    for fold, (train_index, test_index) in enumerate(kf.split(train_record_set)):
        np.random.seed(args.seed + fold)
        indices = np.arange(len(train_index))
        np.random.shuffle(indices)
        split = int(args.finetune_label_ratio * len(indices))
        valid_index = train_index[indices[split:]]
        train_index = train_index[indices[:split]]
        
        train_index = train_index.squeeze()
        valid_index = valid_index.squeeze()
        test_index = test_index.squeeze()
        
        tr_records = torch.from_numpy(train_record_set[train_index]).float()
        tr_labels = torch.from_numpy(train_label_set[train_index]).float()
        val_records = torch.from_numpy(train_record_set[valid_index]).float()
        val_labels = torch.from_numpy(train_label_set[valid_index]).float()
        te_records = torch.from_numpy(train_record_set[test_index]).float()
        te_labels = torch.from_numpy(train_label_set[test_index]).float()

        if load_to_device and device is not None:
            tr_records = tr_records.to(device)
            tr_labels = tr_labels.to(device)
            val_records = val_records.to(device)
            val_labels = val_labels.to(device)
            te_records = te_records.to(device)
            te_labels = te_labels.to(device)

        torch_dataset_Ltrain = Data.TensorDataset(tr_records, tr_labels)
        torch_dataset_valid = Data.TensorDataset(val_records, val_labels)
        torch_dataset_test = Data.TensorDataset(te_records, te_labels)
        fold_datasets.append((torch_dataset_Ltrain, torch_dataset_valid, torch_dataset_test))

   
    os.chdir(args.root)
    return fold_datasets

def file_name(file_dir, file_class):  
    L = []  
    for root, dirs, files in os.walk(file_dir): 
        for file in files: 
            if os.path.splitext(file)[1] == file_class: 
                L.append(os.path.join(root, file)) 
    return L

def conut_nums(dataset_name, csv_file):
    if dataset_name == 'WFDB_PTBXL':
        column_index = 7
    elif dataset_name == 'WFDB_Ga':
        column_index = 8
    elif dataset_name == 'WFDB_Ningbo':
        column_index = 10
    else:
        column_index = 9
    count = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) > column_index:
                count.append(row[column_index])
    count = count[1:]
    count = [int(item) for item in count]
    return count

def multi_label_converter_sepe(multi_label, final_label_list, final_count):
    # 生成严格的 0/1 向量 
    final_count = np.array(final_count)
    num_class = len(final_label_list)
    one_hot_label = np.zeros(num_class)
    for i in multi_label:
        if i in final_label_list:
            one_hot_label[final_label_list.index(i)] = 1
    return one_hot_label, final_count

def load_dataset_super_sepe(dataset_name, root_path, max_length=6144, Norm_type='channel'):
    # 加载预处理配置 
    preprocess_cfg_path = os.path.join(root_path, "L-BiTMix-main", "preprocess.json")
    if not os.path.exists(preprocess_cfg_path):
        # 兼容当前目录
        preprocess_cfg_path = "preprocess.json"
        
    preprocess_cfg = PreprocessConfig(preprocess_cfg_path)
    csv_file = os.path.join(root_path, 'L-BiTMix-main', 'label_mapping.csv') 
    if not os.path.exists(csv_file):
        csv_file = 'label_mapping.csv'
        
    column_index = 1
    data_list = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) > column_index:
                data_list.append(row[column_index])
    data_list = data_list[1:]
    count = conut_nums(dataset_name, csv_file)
    
    # 筛选样本数大于 200 的有效类别
    final_label_list = [label for (label, num) in zip(data_list, count) if num > 200]
    final_count = [num for (label, num) in zip(data_list, count) if num > 200]
    
    # 动态构建原始数据路径
    target_path = os.path.join(root_path, 'raw_data', dataset_name)
    print(f" 正在读取目录: {target_path}")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"找不到数据集文件夹: {target_path}")
        
    os.chdir(target_path)
    file_list_head = sorted(file_name(os.getcwd(), '.hea'))
    file_list_record = sorted(file_name(os.getcwd(), '.mat'))
    print(f" 找到 {len(file_list_head)} 个头文件, {len(file_list_record)} 个记录文件")
    
    record_list = []
    label_list = []
    
    for i in zip(file_list_record, file_list_head):
        file_name_mat, file_name_head = i[0], i[1]
        multi_label = get_labels(load_header(file_name_head))
        one_hot_label, count = multi_label_converter_sepe(multi_label, final_label_list, final_count)
        
        # 过滤全 0 标签
        if np.sum(one_hot_label) == 0:
            continue
            
        # 调用 preprocess.py 中的函数执行滤波清洗
        if Norm_type == 'channel':
            record = preprocess_signal(recording_normalize(file_name_mat, file_name_head), preprocess_cfg,
                                       get_frequency(load_header(file_name_head)), max_length)
        else:
            record = recording_normalize(file_name_mat, file_name_head)
            
        # 确保空间维度一致性
        if record.shape[1] < max_length:
            record = np.column_stack((record, np.zeros((12, max_length - record.shape[1]))))
        elif record.shape[1] > max_length:
            record = record[:, 0:max_length]
            
        record = record.astype('float32')
        record_list.append(record.reshape((record.shape[0], 1, record.shape[1])))
        label_list.append(one_hot_label)
        
    print(f' 数据集 {dataset_name} 解析完成, 最终提取类别数: {len(count)}')
    os.chdir(root_path) # 恢复根目录
    return record_list, label_list

def dataset_organize(args): 
    dataset_list = [args.finetune_dataset]
    
    save_dir = os.path.join(args.root, 'Preprocessed_dataset')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for test_dataset_name in dataset_list:
        print(f" 开始处理数据集: {test_dataset_name}")
        record_list, label_list = load_dataset_super_sepe(dataset_name=test_dataset_name, root_path=args.root)
        
        test_record_set = np.stack(record_list, axis=0)
        test_label_set = np.vstack(label_list)
        num_of_class = str(test_label_set.shape[1])
        
        os.chdir(save_dir)
        output_file = 'class_sepe' + num_of_class + '_dataset_' + test_dataset_name + '_' + '32.hdf5'
        hf = h5py.File(output_file, 'w')
        hf.create_dataset('record_set', data=test_record_set)
        hf.create_dataset('label_set', data=test_label_set)
        print(f" 成功保存 HDF5 缓存文件，张量维度: {test_record_set.shape}, 标签维度: {test_label_set.shape}")
        hf.close()
        
        del record_list, label_list
    os.chdir(args.root)

# 执行入口

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=r'E:\Train', help='你的项目根目录')
    parser.add_argument('--finetune_dataset', type=str, default='WFDB_ChapmanShaoxing', help='要转换的数据集名称')
    args = parser.parse_args()
    
    print(f"启动数据清洗与转换管线...\n目标数据集: {args.finetune_dataset}")
    try:
        dataset_organize(args)
        print("\n 全部数据预处理完成！")
    except Exception as e:
        print(f"\n 运行过程中出现错误: {e}")