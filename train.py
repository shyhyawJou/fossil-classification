import argparse
import os
from time import time
import numpy as np
import random

import torch
from torchvision import transforms as T

from utils import load_data, train



def get_arg():
    arg = argparse.ArgumentParser()
    arg.add_argument('-data', default='fossil', help='root of dataset')
    arg.add_argument('-bs', default=64, type=int, help='batch size')
    arg.add_argument('-lr', type=float, help='initial lr')
    arg.add_argument('-end_lr', type=float, help='when lr < end_lr, end training')
    arg.add_argument('-epoch', type=int, default=150, help='epochs for training')
    arg.add_argument('-save_dir', default='weight', help='location where model will be saved')
    arg.add_argument('-worker', default=4, type=int, help='num of workers')
    arg.add_argument('-seed', type=int, default=None, help='torch random seed')
    arg = arg.parse_args()
    return arg
    

def main():
    arg = get_arg()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(arg.save_dir):
        os.makedirs(arg.save_dir, exist_ok=True) 
        
    if arg.seed is not None:
        random.seed(arg.seed)
        np.random.seed(arg.seed)
        torch.manual_seed(arg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # transforms
    tr_T = T.Compose([T.Resize((224, 224)),
                      T.RandomHorizontalFlip(p=0.5),
                      T.RandomAffine(0, translate=(0.1, 0.1)),
                      T.ToTensor()])
    val_T = T.Compose([T.ToTensor()])

    # load dataset, model, optimizer, lr scheduler
    ds = load_data(arg.data, tr_T, val_T, arg.bs, arg.worker)   

    print()
    print('='*50)
    print('model weight will be saved to:', arg.save_dir)
    print('device:', device)
    print('train num:', len(ds['origin']['train'].dataset))
    print('val num:', len(ds['origin']['val'].dataset))
    print('test num:', len(ds['origin']['test'].dataset))
    print('single data shape:', ds['origin']['train'].dataset[0][0].shape)
    print('='*50)
    print('\nstart training...\n')

    t0 = time()
    result = train(ds, arg.lr, arg.end_lr, device, arg.epoch, arg.save_dir)
    t1 = time()
    
    print()
    print('*'*50)
    print(f'val acc (origin): {result["origin"]:.4f}')
    print(f'val acc (gray): {result["gray"]:.4f}')
    print(f'val acc (skeleton): {result["skeleton"]:.4f}')
    print(f'train time: {t1 - t0:.2f} s')
    print('*'*50)

    

if __name__ == '__main__':
    main()
