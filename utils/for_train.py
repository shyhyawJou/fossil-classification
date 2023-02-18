import pandas as pd

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .for_eval import eval_acc
from .model import load_model



def train(ds, lr, end_lr, device, n_epoch, save_dir):
    n_data, n_batch = len(ds['origin']['train'].dataset), len(ds['origin']['train'])
    loss_fn = nn.CrossEntropyLoss()
    result = {}

    for target in ds:
        # reset
        model = load_model(device)
        opt = SGD(model.parameters(), lr, 0.9, nesterov=True)
        lr_schdr = ReduceLROnPlateau(opt, factor=0.1, patience=2, verbose=True)
        tr_loss_h, val_loss_h = History('min'), History('min')
        tr_acc_h, val_acc_h = History('max'), History('max')
        print()
        print('*'*20, f'training {target} ...', '*'*20)
        
        # start train
        for epoch in range(1, n_epoch + 1):
            print(f'\nEpoch {epoch}:')
            print('-' * 10)
            trace, loss = 0, 0. 
            model.train()
            for i, (x, y) in enumerate(ds[target]['train'], 1):
                opt.zero_grad()
                x, y = x.to(device), y.to(device)           
                logit = model(x)
                batch_loss = loss_fn(logit, y)
                batch_loss.backward()
                opt.step()
                
                with torch.no_grad():
                    trace += torch.sum(logit.max(1)[1] == y)
                    loss += batch_loss * y.numel()
                
                print(f'train: {i}/{n_batch}', end='\r')
            
            loss, acc = loss / n_data, trace / n_data
            tr_acc_h.add(acc)
            tr_loss_h.add(loss)       
            print(f"    tr  loss: {loss:.4f}      tr  acc: {acc:.4f}")
            print(f"min tr  loss: {tr_loss_h.best:.4f}  max tr  acc: {tr_acc_h.best:.4f}")
            
            # validate
            model.eval()
            val_loss, val_acc = eval_acc(model, ds[target]['val'], loss_fn, device)
            val_acc_h.add(val_acc)
            val_loss_h.add(val_loss)
            print(f"    val loss: {val_loss:.4f}      val acc: {val_acc:.4f}")
            print(f"min val loss: {val_loss_h.best:.4f}  max val acc: {val_acc_h.best:.4f}")   
            print(f"lr: {opt.param_groups[0]['lr'] : .6f}")
            print(f'\nBest val acc has not changed for {val_acc_h.n_no_better} epochs')

            # save
            if val_acc_h.better:
                torch.save(model, f'{save_dir}/md_{target}.pt')
                torch.save(model.state_dict(), f'{save_dir}/wt_{target}.pt')
            elif val_acc_h.value == val_acc_h.best and val_loss_h.better:
                torch.save(model, f'{save_dir}/md_{target}.pt')
                torch.save(model.state_dict(), f'{save_dir}/wt_{target}.pt')

            # scheduler
            lr_schdr.step(loss)

            # early stop
            if opt.param_groups[0]['lr'] < end_lr: 
                print('early stop!')
                print(f'end training {target} !!!')
                break
        
        result[target] = val_acc_h.best
        df = pd.DataFrame(zip(tr_acc_h.history, val_acc_h.history),
                          columns=['train', 'val'])
        df.to_excel(f'{save_dir}/{target}_acc.xlsx', index=False)

    return result

    
class History:
    def __init__(self, target='min'):
        self.value = None
        self.best = float('inf') if target == 'min' else 0.
        self.n_no_better = 0
        self.better = False
        self.target = target
        self.history = [] 
        self._check(target)
        
    def add(self, value):
        value = value.item()
        
        if self.target == 'min' and value < self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        elif self.target == 'max' and value > self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        else:
            self.n_no_better += 1
            self.better = False
            
        self.value = value
        self.history.append(value)
        
    def _check(self, target):
        if target not in {'min', 'max'}:
            raise ValueError('target only allow "max" or "min" !')
    
    
