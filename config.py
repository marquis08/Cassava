import os
import cv2
import albumentations as A
abs_path = os.path.dirname(__file__)

import psutil
n_jobs = psutil.cpu_count()


args = {
    "SEED":42,
    # "first_holdout":0.5,
    "n_folds":2,
    "epochs":100, 
    "num_classes":5,
    # "input_size":512,
    "batch_size":10,
    "infer_batch_size":64,
    "infer_model_path":'',
    "infer_best_model_name":'',
    "num_workers":n_jobs,
    "model":"tf_efficientnet_b5_ns", # tf_efficientnet_b2_ns, vit_base_patch16_384, vit_small_patch16_224 "vit_small_patch16_224"
    "loss_fn": "Label", # SCE, CE, Label, BTLL
    "optimizer":"AdamW", # Adam, RAdam, AdamW, SGD
    "scheduler":"WarmupV2", # Cosine, Steplr, Lambda, Plateau, WarmupV2, CosWarm
    "lr":0.00005,   # 0.00025
    "weight_decay":0.0,
    # "center_pad":True,
    # "train_augments":'random_crop, horizontal_flip, vertical_flip, random_rotate, random_grayscale',
    # "valid_augments":'horizontal_flip, vertical_flip',
    "augment_ratio":0.5,
    # "masking_type":None,
    # "max_mask_num":3,
    # "label_smoothing":True,
    "label_smoothing_ratio":0.1,
    "pretrained":True,
    "lookahead":False,
    # "k_param":5,
    # "alpha_param":0.5,
    "patience":3,
    "albu": False,
    # "clahe":True,
    "fp16": True,
    "grayscale": False,
    "DEBUG":True
}

args['tr_aug'] = A.Compose([
                A.RandomResizedCrop(512, 512),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ),

                ], p=1.)

args['val_aug'] = A.Compose([
#                 A.CenterCrop(INPUT_SIZE, INPUT_SIZE, p=1.),
                A.Resize(512, 512),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                )
                ], p=1.)
