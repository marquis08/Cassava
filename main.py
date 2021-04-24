from conf import *

import gc
import os 
import argparse
import sys
import time
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.nn as nn
from torchvision import transforms

from dataloader import *
from models import *
from trainer import *
# from transforms import * # Not using 
from optimizer import *
from utils import seed_everything, find_th, LabelSmoothingLoss
from losses import *

import warnings
from warmup_scheduler import GradualWarmupScheduler
warnings.filterwarnings('ignore')

from glob import glob

def main():
        
    for fold in range(args.n_folds):
        print("\n ##### Fold {} Training ......... #####".format(fold))
        seed_everything(args.SEED)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("\n device", device)
        model = build_model(args, device)
        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer)

        if args.multi_loss:
            criterion = fetch_multiloss(args) # criterion = dict of losses
        else:
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
                    }
        train(args, trn_cfg)
        del model, optimizer, scheduler, criterion, trn_cfg, train_loader, valid_loader
    


if __name__ == '__main__':
    print(args)
    main()
