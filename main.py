from conf import *

import gc
import os 
import argparse
import sys
import time
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.nn as nn
from torchvision import transforms

from dataloader import *
from models import *
from trainer import *
from transforms import *
from optimizer import *
from utils import seed_everything, find_th, LabelSmoothingLoss
from losses import fetch_loss

import warnings
from warmup_scheduler import GradualWarmupScheduler
warnings.filterwarnings('ignore')

from glob import glob

def main():

    # fix seed for train reproduction
    seed_everything(args.SEED)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("\n device", device)


    # TODO dataset loading
    train_df = pd.read_csv('../leaf_merge/clean_df.csv')
    # train_df = train_df.sample(frac=1-args.first_holdout, random_state=42).reset_index(drop=True)
    # train_df = train_df.dropna().reset_index(drop=True)
    # test_df = pd.read_csv('/DATA/testset-for_user.csv', header=None)
    print('train_df shape : ', train_df.shape)
    
    # print(train_df.head())
    # print(train_df.isnull().sum())
    from sklearn.model_selection import train_test_split
    image_path = '../leaf_merge/train_images/train_images/'
    # print(train_df)
    train_df['image_path'] = [os.path.join(image_path, x) for x in train_df['image_id'].values]
    
    # skf_labels = train_df['patient'] + '_' + train_df['label']
   
    # unique_idx = train_df[train_df['count']==1].index
    # non_unique_idx = train_df[train_df['count']>1].index
    trn_idx, val_idx, trn_labels, val_labels = train_test_split(train_df['image_path'].index, train_df['label'].values, 
                                                                test_size=0.05, 
                                                                random_state=0, 
                                                                shuffle=True, 
                                                                stratify=train_df['label'].values)
    
    # print(train_df.head())
    # valid set define
    trn_image_paths = train_df.loc[trn_idx, 'image_path'].values
    val_image_paths = train_df.loc[val_idx, 'image_path'].values
    
    print('\n')
    print('train, valid split 0.05 ', len(trn_image_paths), len(trn_labels), len(val_image_paths), len(val_labels))
    print('\n')
    # print(trn_image_paths[:5], trn_labels[:5])
    # print(val_image_paths[:5], val_labels[:5])

    # valid_transforms = create_val_transforms(args, args.input_size)
    # valid_transforms = args.val_aug
    if args.DEBUG:
        valid_dataset = LeafDataset(args, val_image_paths[:100], val_labels[:100], args.val_aug, is_test=False)
    else:
        valid_dataset = LeafDataset(args, val_image_paths, val_labels, args.val_aug, is_test=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    
    # trn_skf_labels = (train_df.loc[trn_idx, 'patient'] + train_df.loc[trn_idx, 'label']).values
    # print('skf labels head : ', trn_skf_labels[:5])

    if args.DEBUG:
        print('\n#################################### DEBUG MODE')
    else:
        print('\n################################### MAIN MODE')
        print(trn_image_paths.shape, trn_labels.shape) 

    # train set define
    train_dataset_dict = {}
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.SEED)
    nsplits = [val_idx for _, val_idx in skf.split(trn_image_paths, trn_labels)]
    #print(nsplits)
    #np.save('nsplits.npy', nsplits)
    
    print('\nload nsplits')
    # nsplits = np.load('nsplits.npy', allow_pickle=True)
    # print(nsplits)

    for idx, val_idx in enumerate(nsplits):#trn_skf_labels
        
        sub_img_paths = np.array(trn_image_paths)[val_idx]
        sub_labels = np.array(trn_labels)[val_idx]

        if args.DEBUG:
            sub_img_paths = sub_img_paths[:200]
            sub_labels = sub_labels[:200] 

        
        # train_transforms = create_train_transforms(args.tr_aug)
        # train_transforms = create_train_transforms(args, args.input_size)
        #train_dataset = LeafDataset(args, sub_img_paths, sub_labels, train_transforms, use_masking=True, is_test=False)
        train_dataset_dict[idx] = [args, sub_img_paths, sub_labels, args.tr_aug]
        print(f'train dataset complete {idx}/{args.n_folds}, ')

    print("number of train datasets: ", len(train_dataset_dict))

    # define model
    model = build_model(args, device)

    # optimizer definition
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 9)
    # scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=1, after_scheduler=scheduler_cosine)

    criterion = fetch_loss(args)
    # if args.label_smoothing:
    #     criterion = LabelSmoothingLoss(classes=args.num_classes, smoothing=args.label_smoothing_ratio)
    # else:
    #     criterion = nn.CrossEntropyLoss()

    trn_cfg = {'train_datasets':train_dataset_dict,
                    'valid_loader':valid_loader,
                    'model':model,
                    'criterion':criterion,
                    'optimizer':optimizer,
                    'scheduler':scheduler,
                    'device':device,
                    'fold_num':0,
                    # 'input_size': args.input_size,
                    'batch_size': args.batch_size,
                    }

    train(args, trn_cfg)


if __name__ == '__main__':
    print(args)
    main()
