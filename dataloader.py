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
        transforms=None, use_masking=False, is_test=False
    ):
        self.args = args
        self.image_paths = image_paths
        self.use_masking = use_masking
        self.labels = labels
        # self.images = [] # use if you have enough mem to store images
        # self.default_transforms = default_transforms
        self.transforms = transforms
        self.is_test = is_test
        
        # print('########################### dataset loader to memory for Faster Loading')
        # for image_path in tqdm(self.image_paths):
        #     if args.grayscale:
        #         # print("THIS IS GRAY")
        #         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #         image = np.expand_dims(image, -1)
        #     else:
                
        #         image = cv2.imread(image_path)
        #         # print("THIS IS 3CH : {}".format(image.shape))
        #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #         # image = Image.fromarray(image)

        #     # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #     #if args.clahe:
        #     #    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
        #     #    image = clahe.apply(image)

        #     self.images.append(image)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # if get image one by one
        # image_path = self.image_paths[index]
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

        if self.transforms:
            image = self.transforms(image=image)['image']

        if self.args.grayscale:
            image = img_to_tensor(image) # this is normalize and to tensor funtion
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
    # train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # train_df = train_df.dropna().reset_index(drop=True)
    # test_df = pd.read_csv('/DATA/testset-for_user.csv', header=None)
    print('train_df shape : ', train_df.shape)
    
    # print(train_df.head())
    # print(train_df.isnull().sum())
    from sklearn.model_selection import train_test_split
    image_path = '../leaf_merge/train_images/train_images/'
    # print(train_df)
    train_df['image_path'] = [os.path.join(image_path, x) for x in train_df['image_id'].values]
    
    # if args.sub_train:
    #     print("Using Sub Train ....................")

    #     # skf_labels = train_df['patient'] + '_' + train_df['label']
    
    #     # unique_idx = train_df[train_df['count']==1].index
    #     # non_unique_idx = train_df[train_df['count']>1].index
    #     trn_idx, val_idx, trn_labels, val_labels = train_test_split(train_df['image_path'].index, train_df['label'].values, 
    #                                                                 test_size=0.05, 
    #                                                                 random_state=0, 
    #                                                                 shuffle=True, 
    #                                                                 stratify=train_df['label'].values)
        
    #     trn_image_paths = train_df.loc[trn_idx, 'image_path'].values
    #     val_image_paths = train_df.loc[val_idx, 'image_path'].values
        
    #     print('\n')
    #     print('train, valid split 0.05 ', len(trn_image_paths), len(trn_labels), len(val_image_paths), len(val_labels))
    #     print('\n')
        
    #     if args.DEBUG:
    #         valid_dataset = LeafDataset(args, val_image_paths[:100], val_labels[:100], args.val_aug, is_test=False)
    #     else:
    #         valid_dataset = LeafDataset(args, val_image_paths, val_labels, args.val_aug, is_test=False)
    #     valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
        
    #     if args.DEBUG:
    #         print('\n#################################### DEBUG MODE')
    #     else:
    #         print('\n################################### MAIN MODE')
    #         print(trn_image_paths.shape, trn_labels.shape) 

    #     # train set define
    #     train_dataset_dict = {}
    #     skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.SEED)
    #     nsplits = [val_idx for _, val_idx in skf.split(trn_image_paths, trn_labels)]
    #     #np.save('nsplits.npy', nsplits)
        
    #     print('\nload nsplits')
    #     # nsplits = np.load('nsplits.npy', allow_pickle=True)

    #     for idx, val_idx in enumerate(nsplits):#trn_skf_labels
            
    #         sub_img_paths = np.array(trn_image_paths)[val_idx]
    #         sub_labels = np.array(trn_labels)[val_idx]

    #         if args.DEBUG:
    #             sub_img_paths = sub_img_paths[:200]
    #             sub_labels = sub_labels[:200] 

    #         # train_transforms = create_train_transforms(args.tr_aug)ize)
    #         #train_dataset = LeafDataset(args, sub_img_paths, sub_labels, train_transforms, use_masking=True, is_test=False)
    #         train_dataset_dict[idx] = [args, sub_img_paths, sub_labels, args.tr_aug]
    #         print(f'train dataset complete {idx}/{args.n_folds}, ')

    #     return train_dataset_dict, valid_loader

    # else:
    print(" Making Dataset ")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.SEED)

    for idx, (train_idx, val_idx) in enumerate(skf.split(X=train_df['image_path'].index, y=train_df['label'])):
        train_df.loc[train_df.iloc[val_idx].index, 'fold'] = idx

    if args.DEBUG:
        print('\n#################################### DEBUG MODE')
        df_train = train_df[train_df['fold']!=fold_num][:200]
        df_valid = train_df[train_df['fold']==fold_num][:200]
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