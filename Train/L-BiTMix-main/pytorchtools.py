import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self,patience=7, verbose=False,dataset_name='Ga',delta=0,args=None,path = '/checkpoint'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.dataset_name=dataset_name
        self.args=args
        self.path = self.args.root + path
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
    
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        os.makedirs(self.path, exist_ok=True)
        fold_idx = getattr(self.args, 'current_fold', 0)
        filename = f"{self.dataset_name}_fold{fold_idx}_checkpoint.pkl"
        save_path = os.path.join(self.path, filename)

        if self.args.ranklist == 'FT':
            torch.save(model.state_dict(), save_path)
        else:
            saving_lora_checkpoint(model, save_path)
        self.val_loss_min = val_loss

def saving_lora_checkpoint(net,path):
    net_state_dict = net.state_dict()
    saved_state_dict = {}
    for name, param in net_state_dict.items():
        if name.find('lora') > -1 or name.find('bias') > -1:
            saved_state_dict[name] = param
        elif name.find('classifier.1.weight') > -1:
            saved_state_dict[name] = param
        elif name.find('bn') > -1 or name.find('norm') > -1:
            saved_state_dict[name] = param
    torch.save(saved_state_dict, path)
    return 0