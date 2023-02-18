import argparse
import os
from PIL import Image
from pathlib import Path as p
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torchvision import transforms as T



def get_arg():
    arg = argparse.ArgumentParser()
    arg.add_argument('-data', default='fossil', help='root of dataset')
    arg.add_argument('-weight', default='weight', help='folder of model')
    arg = arg.parse_args()
    return arg
    

def main():
    arg = get_arg()

    classes = sorted(os.listdir(f'{arg.data}/origin/test'))
    cls_to_lb = dict(zip(classes, range(len(classes))))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    preproc = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    model = {'origin': torch.load(f'{arg.weight}/md_origin.pt', device).eval(),
             'gray': torch.load(f'{arg.weight}/md_gray.pt', device).eval(),
             'skeleton': torch.load(f'{arg.weight}/md_skeleton.pt', device).eval()}

    paths = list(p(f'{arg.data}/origin/test').glob('**/*.png'))
    correct = {'origin': 0, 'OGS': 0}
    for i, origin_path in enumerate(tqdm(paths), 1):
        gray_path = f'{arg.data}/gray/test/{"/".join(origin_path.parts[-2:])}'
        skeleton_path = f'{arg.data}/skeleton/test/{"/".join(origin_path.parts[-2:])}'
         
        origin = preproc(Image.open(origin_path).convert('RGB'))[None]
        gray = preproc(Image.open(gray_path).convert('RGB'))[None]
        skeleton = preproc(Image.open(skeleton_path).convert('RGB'))[None]
        origin, gray, skeleton = origin.to(device), gray.to(device), skeleton.to(device)

        conf_origin = F.softmax(model['origin'](origin), 1)
        conf_gray = F.softmax(model['gray'](gray), 1)
        conf_skeleton = F.softmax(model['skeleton'](skeleton), 1)

        # soft vote
        label = cls_to_lb[origin_path.parts[-2]]
        if conf_origin.max(1)[1] == label:
            correct['origin'] += 1
        if ((conf_origin + conf_gray + conf_skeleton) / 3).max(1)[1] == label:        
            correct['OGS'] += 1

    print()
    print('*'*50)
    print('total img:', i)
    print(f'test acc (origin): {correct["origin"] / i:.4f}')
    print(f'test acc (OGS): {correct["OGS"] / i:.4f}')
    print('*'*50)

    

if __name__ == '__main__':
    main()
