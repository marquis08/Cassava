from conf import *

import os
import cv2
import copy
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageFile, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.functional import img_to_tensor
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold

ImageFile.LOAD_TRUNCATED_IMAGES = True 
# If truncated image is reduce to np array, 
# pil produces error. To prevent, turn True on this option
# https://deep-deep-deep.tistory.com/34

class LeafDataset(Dataset):
    def __init__(self, args, image_paths, labels=None, 
        transforms=None, is_test=False
    ):
        self.args = args
        self.image_paths = image_paths
        self.labels = labels
        # self.images = [] # use if you have enough mem to s
        self.transforms = transforms
        self.is_test = is_test
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        if self.args.backend == "pil":
            if self.args.grayscale:
                image = Image.open(self.image_paths[index]).convert("L")
                image = np.array(image)
                image = np.expand_dims(image, -1)
            else:    
                image = Image.open(self.image_paths[index])
                image = np.array(image)
        elif self.args.backend == "cv2":
            if self.args.grayscale:
                image = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
                image = np.expand_dims(image, -1)
            else:
                image = cv2.imread(self.image_paths[index])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise Exception("Backend not implemented")

        # Applying Albumentation augments
        if self.transforms:
            image = self.transforms(image=image)['image']

        # Normalize and to Tensor
        if self.args.grayscale:
            image = img_to_tensor(image) 
        else:
            image = img_to_tensor(image, {"mean": [0.485, 0.456, 0.406],
                                   "std": [0.229, 0.224, 0.225]})
        # image = torch.from_numpy(image.transpose((2, 0, 1)))
        # image = np.transpose(image, (2,0,1)).astype(np.float32)


        if self.is_test:
            return image
        else:
            label = self.labels[index]
            return image, label#torch.tensor(img, dtype=torch.float32),\
                 #torch.tensor(self.labels[index], dtype=torch.long)

def get_df(args, fold_num):
    # TODO dataset loading
    train_df = pd.read_csv('../leaf_merge/clean_df.csv')
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print('train_df shape : ', train_df.shape)
    
    from sklearn.model_selection import train_test_split
    image_path = '../leaf_merge/train_images/train_images/'
    train_df['image_path'] = [os.path.join(image_path, x) for x in train_df['image_id'].values]
    

    print(" Making Dataset ")
    skf = StratifiedKFold(n_splits=args.n_folds)

    for idx, (train_idx, val_idx) in enumerate(skf.split(X=train_df['image_path'].index, y=train_df['label'])):
        train_df.loc[train_df.iloc[val_idx].index, 'fold'] = idx

    # skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.SEED)

    # for idx, (train_idx, val_idx) in enumerate(skf.split(X=train_df['image_path'].index, y=train_df['label'])):
    #     train_df.loc[train_df.iloc[val_idx].index, 'fold'] = idx

    if args.DEBUG:
        print('\n#################################### DEBUG MODE')
        df_train = train_df[train_df['fold']!=fold_num][:100]
        df_valid = train_df[train_df['fold']==fold_num][:100]
        print(df_train.shape, df_valid.shape) 
    else:
        print('\n################################### MAIN MODE')
        df_train = train_df[train_df['fold']!=fold_num]
        df_valid = train_df[train_df['fold']==fold_num]
        print(df_train.shape, df_valid.shape) 
    
    train_image_paths = [os.path.join(image_path, x) for x in df_train.image_id.values]
    valid_image_paths = [os.path.join(image_path, x) for x in df_valid.image_id.values]
    train_targets = df_train.label.values
    valid_targets = df_valid.label.values
    
    train_dataset = LeafDataset(args, train_image_paths, train_targets, args.tr_aug, is_test=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.tr_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    valid_dataset = LeafDataset(args, valid_image_paths, valid_targets, args.val_aug, is_test=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader