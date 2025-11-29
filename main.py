from pipeline_ecg import ECG_TCN_compa,MixTcn_Lite
import os
import argparse
import numpy as np
import warnings
import sys
import torch
import gc
def ECG_config(seed,root,op_config='HM_BiTCN'):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task', type=str, default='within')
    parser.add_argument('--model_config', type=str, default='medium')
    parser.add_argument('--semi_config', type=str, default='default')

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--ranklist', type=str, default='lora_ave')
    parser.add_argument('--r', type=int, default=0)  # LoRA rank：[4, 8, 16]
    parser.add_argument('--root', type=str, default=root)
    parser.add_argument('--seed', type=int, default=seed)

    parser.add_argument('--finetune_epoch', type=int, default=200)
    parser.add_argument('--num_class', type=int, default=25)
    parser.add_argument('--finetune_label_ratio', type=float, default=0.90)
    parser.add_argument('--pretrain_dataset', type=str, default='CODE_test')
    parser.add_argument('--finetune_dataset', type=str, default='WFDB_ChapmanShaoxing')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--device_train',type=str,default='cuda:0')
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--op_config', type=str, default=op_config)

    ##hyper-parameter for HM-BiTCN
    parser.add_argument('--mix_weight', type=float, default=0.3)
    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--leads', type=int, default=12)

    parser.add_argument('--enable_amp', type=bool, default=True)
    parser.add_argument('--load_data_to_gpu', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)    

    # ## hyper-parameter for hybrid_tuning
    # parser.add_argument('--q', type=int, default=1)
    # parser.add_argument('--bp_batch', type=int, default=2)
    # parser.add_argument('--zo_eps', type=float, default=1e-3)
    # parser.add_argument('--trainer', type=str, default='zo')  # zo_sign_opt
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    # parser.add_argument('--gradient_sparsity', type=float, default=None)
    # parser.add_argument('--perturbation_mode', type=str, default='two_side')
    # parser.add_argument('--coef', type=float, default=0.85)
    # parser.add_argument('--no_grad_correct', type=bool, default=False)
    # ## hyper-parameter for know distillation
    # parser.add_argument('--enable_distillation', type=bool, default=True)
    # parser.add_argument('--student_load_pretrain', type=bool, default=True)
    # parser.add_argument('--leads_for_teacher', type=int, default=12)
    # parser.add_argument('--leads_for_student', type=int, default=12)
    # parser.add_argument('--tuning', type=str, default='H-tuning')
    args = parser.parse_args()
    return args

def exp_main_search(args):
    """
    mix_weight controls the contribution of HM block"""
    print("!!HM_BiTCN Search mix_weight List!!")
    w_list = [0.1, 0.3, 0.5, 0.7, 1]
    args.ranklist = 'FT'
    dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_ChapmanShaoxing']
    num_class_list = [18, 19, 16]
    save_file_name = 'HM_BiTCN_Search_MixWeight.npy'
    print(save_file_name)

    os.chdir(args.root + '/result')
    if os.path.exists(save_file_name):
        print('file exist')
        result = np.load(save_file_name, allow_pickle=True).tolist()
    else:
        result = []

    start_index = len(result)
    if start_index >= len(w_list):
        return
    
    for j in range(start_index, len(w_list)):
        args.mix_weight = w_list[j]
        print('current mix_weight', args.mix_weight)
        result_dataset = []
        for i in range(3):
            args.finetune_dataset = dataset_list[i]
            args.num_class = num_class_list[i]
            print('TyT->',args.finetune_dataset)
            result_dataset.append(MixTcn_Lite(args=args))
        result.append(result_dataset)
        print('running progress', len(result))
        os.chdir(args.root + '/result')
        np.save(save_file_name, result)

def exp_main_MixTcnLite(args):
    print("!!HM_BiTCN in !!" + str(args.op_config))
    args.ranklist = 'FT'
    dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_ChapmanShaoxing']
    num_class_list = [18, 19, 16]
    save_file_name = 'CM_biTCN_Results_' + str(args.op_config) + '.npy'
    print(save_file_name)
    
    os.chdir(args.root + '/result')
    if os.path.exists(save_file_name):
        print('file exist')
        result = np.load(save_file_name, allow_pickle=True).tolist()
        start = len(result)
    else:
        result = []
        start = 0

    print('current seed', args.seed)
    print("learning rate:", args.learning_rate)
    print("mix weight:", args.mix_weight)
    for j in range(start, 3):
        args.finetune_dataset = dataset_list[j]
        args.num_class = num_class_list[j]
        print('TyT->',args.finetune_dataset)
        result.append(MixTcn_Lite(args=args))
        os.chdir(args.root + '/result')
        np.save(save_file_name, result) 

# def exp_main_MixTcnLite_replace(args):
#     print("!!Replace HM_BiTCN!!")
#     args.ranklist = 'FT'
#     dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_ChapmanShaoxing']
#     num_class_list = [18, 19, 16]
#     save_file_name = 'CM_biTCN_Results_replace.npy'
#     print(save_file_name)

#     os.chdir(args.root + '/result')
#     if os.path.exists(save_file_name):
#         print('file exist')
#         result = np.load(save_file_name, allow_pickle=True).tolist()
#     else:
#         result = []

#     print('current r', args.r)
#     print('current seed', args.seed)
#     print("learning rate:", args.learning_rate)
#     print("mix weight:", args.mix_weight)
#     for j in range(3):
#         args.finetune_dataset = dataset_list[j]
#         args.num_class = num_class_list[j]
#         print('TyT->',args.finetune_dataset)
#         result.append(MixTcn_Lite_replace (args=args))
#         os.chdir(args.root + '/result')
#         np.save(save_file_name, result)
        
# def exp_main_MixTcnLite_parallel(args):
#     print("!!HM_BiTCN in parallel!!")
#     args.mix_weight = 0.5
#     args.ranklist = 'FT'
#     dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_ChapmanShaoxing']
#     num_class_list = [18, 19, 16]
#     save_file_name = 'CM_biTCN_Results_parallel.npy'
#     print(save_file_name)

#     os.chdir(args.root + '/result')
#     if os.path.exists(save_file_name):
#         print('file exist')
#         result = np.load(save_file_name, allow_pickle=True).tolist()
#     else:
#         result = []

#     print('current r', args.r)
#     print('current seed', args.seed)
#     print("learning rate:", args.learning_rate)
#     print("mix weight:", args.mix_weight)
    
#     for j in range(3):
#         args.finetune_dataset = dataset_list[j]
#         args.num_class = num_class_list[j]
#         print('TyT->',args.finetune_dataset)
#         result.append(MixTcn_Lite_parallel(args=args))
#         os.chdir(args.root + '/result')
#         np.save(save_file_name, result)
        
# def exp_main_only_HM_BiTCN(args):
#     print("!!Only HM_BiTCN!!")
#     args.ranklist = 'FT'
#     dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_ChapmanShaoxing']
#     num_class_list = [18, 19, 16]
   
#     save_file_name = 'CM_biTCN_Results_only_HM_BiTCN.npy'
#     print(save_file_name)

#     os.chdir(args.root + '/result')
#     if os.path.exists(save_file_name):
#         print('file exist')
#         result = np.load(save_file_name, allow_pickle=True).tolist()
#     else:
#         result = []

#     print('current r', args.r)
#     print('current seed', args.seed)
#     print("learning rate:", args.learning_rate)
#     print("mix weight:", args.mix_weight)
#     for j in range(3):
#         args.finetune_dataset = dataset_list[j]
#         args.num_class = num_class_list[j]
#         print('TyT->',args.finetune_dataset)
#         result.append(MixTcn_Lite_only_HM_BiTCN(args=args))
#         os.chdir(args.root + '/result')
#         np.save(save_file_name, result)

# def exp_main_parallel_wo_HM(args):
#     print("!!Ablution without HM!!")
#     args.ranklist = 'FT'
#     dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_ChapmanShaoxing']
#     num_class_list = [18, 19, 16]
#     save_file_name = 'CM_biTCN_Results_wo.npy'
#     print(save_file_name)

#     os.chdir(args.root + '/result')
#     if os.path.exists(save_file_name):
#         print('file exist')
#         result = np.load(save_file_name, allow_pickle=True).tolist()
#     else:
#         result = []

#     print('current r', args.r)
#     print('current seed', args.seed)
#     print("learning rate:", args.learning_rate)
#     print("mix weight:", args.mix_weight)
#     for j in range(3):
#         args.finetune_dataset = dataset_list[j]
#         args.num_class = num_class_list[j]
#         print('TyT->',args.finetune_dataset)
#         result.append(Ablution_Module(args=args))
#         os.chdir(args.root + '/result')
#         np.save(save_file_name, result)
    
def exp_main_comparation(args):
    print("!!HM_BiTCN_comparation!!")
    args.ranklist = 'FT'
    dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_ChapmanShaoxing']
    num_class_list = [18, 19, 16]
    save_file_name = 'CM_biTCN_Results_comparation.npy'
    print(save_file_name)

    os.chdir(args.root + '/result')
    if os.path.exists(save_file_name):
        print('file exist')
        result = np.load(save_file_name, allow_pickle=True).tolist()
        start = len(result)
    else:
        result = []
        start = 0

    print('current seed', args.seed)
    print("learning rate:", args.learning_rate)
    
    for j in range(start, 3):
        args.finetune_dataset = dataset_list[j]
        args.num_class = num_class_list[j]
        print('TyT->',args.finetune_dataset)
        result.append(ECG_TCN_compa(args=args))
        os.chdir(args.root + '/result')
        np.save(save_file_name, result) 
     
def Task_ECG_MHC(seed, root, op_config='MHC'):
    args = ECG_config(seed, root, op_config)
    args.finetune_dataset = args.finetune_dataset
    args.model_config = 'medium'
    args.semi_config = 'nosemi'
    print('seed:', args.seed)
    print('device:', args.device)
    print('model_config:', args.model_config)
    # if args.op_config == 'HM_BiTCN_series':
    #     exp_main_MixTcnLite_series(args)
    # elif args.op_config == 'HM_BiTCN_parallel':
    #     exp_main_MixTcnLite_parallel(args)
    # elif args.op_config == 'HM_BiTCN_replace':
    #     exp_main_MixTcnLite_replace(args)
    # elif args.op_config == 'only_HM_BiTCN':
    #     exp_main_only_HM_BiTCN(args)
    # elif args.op_config == 'HM_BiTCN_ablution':
    #     exp_main_parallel_wo_HM(args)
    # elif args.op_config == 'HM_BiTCN_search':
    #     exp_main_search(args)
    if args.op_config == 'HM_BiTCN_search':
        exp_main_search(args)
    elif args.op_config == 'HM_BiTCN_TCN_COMPA':
        exp_main_comparation(args)
    else:
        exp_main_MixTcnLite(args)
    
if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
    warnings.filterwarnings("ignore", category=FutureWarning)
    root = os.getcwd()
    # root = os.getcwd() + "/zy/CSCWDcode"
    # 从原始数据生成.hdf5预处理数据集
    # args = ECG_config(18, root, '')
    # dataset_organize(args)
    # 实验入口
    seed = 18   
    Task_ECG_MHC(seed, root, 'HM_BiTCN_search')
    Task_ECG_MHC(seed, root, 'HM_BiTCN_series')
    Task_ECG_MHC(seed, root, 'HM_BiTCN_TCN_COMPA')
    Task_ECG_MHC(seed, root, 'HM_BiTCN_ablution')
    Task_ECG_MHC(seed, root, 'HM_BiTCN_replace')
    Task_ECG_MHC(seed, root, 'only_HM_BiTCN')
    