import os
import cv2
import albumentations as A
abs_path = os.path.dirname(__file__)

import psutil
n_jobs = psutil.cpu_count()


args = {
    "SEED":42,
    # "sub_train":False,
    "n_folds":5,
    "epochs":1, 
    "num_classes":5,
    # "input_size":512,
    "tr_batch_size":8,
    "val_batch_size":32,
    # "infer_batch_size":64,
    # "infer_model_path":'',
    # "infer_best_model_name":'',
    "backend":"pil", # pil, cv2
    "num_workers":n_jobs,
    "timm":True,
    "timm_model":"tf_efficientnet_b4_ns", # tf_efficientnet_b4_ns, tf_efficientnet_b2_ns, vit_base_patch16_384, vit_small_patch16_224 "vit_small_patch16_224"
    "model":"efficientnet-b4",
    "loss_fn": "Label", # SCE, CE, Label, BTLL
    "optimizer":"RAdam", # Adam, RAdam, AdamW, SGD
    "scheduler":"CosWarm", # Cosine, Steplr, Lambda, Plateau, WarmupV2, CosWarm
    "lr":3e-4,   # 0.00025
    "weight_decay":1e-4,
    # "center_pad":True,
    # "train_augments":'random_crop, horizontal_flip, vertical_flip, random_rotate, random_grayscale',
    # "valid_augments":'horizontal_flip, vertical_flip',
    "augment_ratio":0.5,
    # "masking_type":None,
    # "max_mask_num":3,
    # "label_smoothing":True,
    "label_smoothing_ratio":0.4,
    "pretrained":True,
    "lookahead":False,
    # "k_param":5,
    # "alpha_param":0.5,
    # "patience":3,
    # "albu": False,
    # "clahe":True,
    "fp16": True,
    "grayscale": True,
    # "dropout_ensemble": False,
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
