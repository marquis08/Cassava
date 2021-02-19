import os
import cv2
import albumentations as A
abs_path = os.path.dirname(__file__)

import psutil
n_jobs = psutil.cpu_count()


args = {
    "SEED":42,
    "n_folds":5,
    "epochs":10, 
    "num_classes":5,
    "tr_batch_size":12,
    "val_batch_size":32,
    "backend":"pil", # pil, cv2
    "num_workers":n_jobs,
    "timm":True,
    "timm_model":"tf_efficientnet_b4_ns", # tf_efficientnet_b4_ns, tf_efficientnet_b2_ns, vit_base_patch16_384, vit_small_patch16_224 "vit_small_patch16_224"
    "model":"efficientnet-b4", # using EfficientNet from efficientnet_pytorch
    "multi_loss": True,
    "multi_loss_list":['CE','Label'],
    "multi_loss_epoch_thr":5,
    "loss_fn": "Label", # SCE, CE, Label, BTLL
    "optimizer":"RAdam", # Adam, RAdam, AdamW, SGD, Lookahead
    "scheduler":"CosWarm", # Cosine, Steplr, Lambda, Plateau, WarmupV2, CosWarm
    "lr":3e-4,   
    "weight_decay":1e-4,
    "augment_ratio":0.5,
    "label_smoothing_ratio":0.4,
    "pretrained":True,
    "fp16": True,
    "grayscale": False,
    "DEBUG":True
}

args['tr_aug'] = A.Compose([
                # A.Resize(512, 512),
                # A.RandomCrop(448, 448, p=0.5),
                # A.RandomResizedCrop(512, 512),
                A.Resize(512, 512),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                # A.OneOf([
                #         A.MotionBlur(blur_limit=5),
                #         A.MedianBlur(blur_limit=5),
                #         #A.GaussianBlur(blur_limit=5),
                #         A.GaussNoise(var_limit=(5.0, 30.0))], p=0.5),
                # A.Normalize(),

                ], p=1.)

args['val_aug'] = A.Compose([
                # A.Resize(512, 512),
                # A.RandomCrop(448, 448, p=0.5),
                # A.RandomResizedCrop(512, 512),
                A.Resize(512, 512),
                # A.Normalize()
                ], p=1.)
