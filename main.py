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

    # # fix seed for train reproduction
    # seed_everything(args.SEED)
    
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print("\n device", device)


    # # train_dataset_dict, valid_loader = get_df(args)

    # # define model
    # model = build_model(args, device)

    # # optimizer definition
    # optimizer = build_optimizer(args, model)
    # scheduler = build_scheduler(args, optimizer)
    # # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 9)
    # # scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=1, after_scheduler=scheduler_cosine)

    # criterion = fetch_loss(args)
    # if args.label_smoothing:
    #     criterion = LabelSmoothingLoss(classes=args.num_classes, smoothing=args.label_smoothing_ratio)
    # else:
    #     criterion = nn.CrossEntropyLoss()

    # if args.sub_train:
    #     train_dataset_dict, valid_loader = get_df(args)
    #     trn_cfg = {'train_datasets':train_dataset_dict,
    #                     'valid_loader':valid_loader,
    #                     'model':model,
    #                     'criterion':criterion,
    #                     'optimizer':optimizer,
    #                     'scheduler':scheduler,
    #                     'device':device,
    #                     'fold_num':0,
    #                     # 'input_size': args.input_size,
    #                     'batch_size': args.batch_size,
    #                     }
    #     train(args, trn_cfg)                        
    # else:
        
    for fold in range(args.n_folds):
        print("\n ##### Fold {} Training ......... #####".format(fold))
        seed_everything(args.SEED)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("\n device", device)
        model = build_model(args, device)
        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer)

        criterion = fetch_loss(args)
        train_loader, valid_loader = get_df(args, fold)
        trn_cfg = {'train_loader':train_loader,
                    'valid_loader':valid_loader,
                    'model':model,
                    'criterion':criterion,
                    'optimizer':optimizer,
                    'scheduler':scheduler,
                    'device':device,
                    'fold_num':fold,
                    # 'input_size': args.input_size,
                    # 'tr_batch_size': args.tr_batch_size,
                    # 'val_batch_size': args.val_batch_size,
                    }
        train(args, trn_cfg)
        del model, optimizer, scheduler, criterion, trn_cfg
    


if __name__ == '__main__':
    print(args)
    main()
