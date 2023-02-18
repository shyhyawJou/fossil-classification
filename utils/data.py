from pathlib import Path as p
import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.morphology import skeletonize
import numpy as np
import cv2
from PIL import Image

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF



CLASS = {'Chusenella', 
         'Nankinella', 
         'Pseudoschwagerina', 
         'Parafusulina', 
         'Quasifusulina', 
         'Fusulinella', 
         'Pseudofusulina', 
         'Rugosofusulina', 
         'Eoparafusulina', 
         'Schwagerina', 
         'Neoschwagerina', 
         'Triticites', 
         'Schubertella', 
         'Eostaffella', 
         'Misellina', 
         'Fusulina'}


def load_data(data_root, tr_T, val_T, batch_size, n_worker):
    ds = {'origin': {}, 'gray': {}, 'skeleton': {}}
    for data_dir in ['origin', 'gray', 'skeleton']:
        tr = ImageFolder(f'{data_root}/{data_dir}/train', tr_T)
        val = ImageFolder(f'{data_root}/{data_dir}/val', val_T)
        test = ImageFolder(f'{data_root}/{data_dir}/test', val_T)
        ds[data_dir]['train'] = DataLoader(tr, 
                                           batch_size, 
                                           shuffle=True,
                                           pin_memory=True, 
                                           num_workers=n_worker)
        ds[data_dir]['val'] = DataLoader(val, 
                                         batch_size, 
                                         shuffle=False,
                                         pin_memory=True, 
                                         num_workers=n_worker)
        ds[data_dir]['test'] = DataLoader(test, 
                                          batch_size, 
                                          shuffle=False,
                                          pin_memory=True, 
                                          num_workers=n_worker)
    return ds


class Split:
    def __init__(self, data_root, save_dir, seed):        
        # check class
        classes = os.listdir(data_root)
        class_label = dict(zip(classes, range(len(classes))))
        assert set(classes) == CLASS, f'Valid classes should be {CLASS} but get {classes}'

        # check number of images
        paths, labels = [], []
        for path in p(data_root).glob('**/*.png'):
            paths.append(path)
            labels.append(class_label[path.parts[-2]])
        assert len(paths) == 2400, f'should be 2400 imgs but get {len(paths)}'

        # split into 110 / 20 / 20
        (tr_paths, test_paths, 
        tr_labels, test_labels) = train_test_split(paths, 
                                                   labels, 
                                                   test_size=320, 
                                                   random_state=seed, 
                                                   stratify=labels)
        tr_paths, val_paths = train_test_split(tr_paths, 
                                               test_size=320, 
                                               random_state=seed, 
                                               stratify=tr_labels)
        
        # make dataset
        self.make_dirs(save_dir)
        for i, paths in enumerate([tr_paths, val_paths, test_paths]):
            for path in tqdm(paths):
                if i == 0:
                    folder = 'train'
                elif i == 1:
                    folder = 'val'
                else:
                    folder = 'test'  

                self.save_origin(path, folder, save_dir)
                self.save_gray_and_skeleton(path, folder, save_dir)

    def save_origin(self, path, folder, save_dir):
        dst = f'{save_dir}/origin/{folder}/{"/".join(path.parts[-2:])}'
        shutil.copyfile(path, dst)

    def save_gray_and_skeleton(self, path, folder, save_dir):
        gray = self.to_gray(cv2.imread(str(path)))
        skeleton = self.skeletonize(gray)
        dst_gray = f'{save_dir}/gray/{folder}/{"/".join(path.parts[-2:])}'
        dst_skeleton = f'{save_dir}/skeleton/{folder}/{"/".join(path.parts[-2:])}'
        cv2.imwrite(dst_gray, gray)
        cv2.imwrite(dst_skeleton, skeleton)

    def make_dirs(self, save_dir):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        for folder1 in ['origin', 'gray', 'skeleton']:
            for folder2 in ['train', 'val', 'test']:
                for cls in CLASS:
                    os.makedirs(f'{save_dir}/{folder1}/{folder2}/{cls}', exist_ok=True)

    def to_gray(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def skeletonize(self, img):
        img = cv2.adaptiveThreshold(img, 
                                    255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 
                                    41, 
                                    5)
        img[img == 255] = 1        
        img = skeletonize(img)
        img = np.uint8(~img) * 255
        return img
    