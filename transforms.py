from conf import *

import math
import random
from PIL import Image, ImageOps
# from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop, RandomApply, Resize, CenterCrop, RandomAffine
# from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomGrayscale, RandomRotation
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from albumentations import DualTransform
# from albumentations import Compose, RandomBrightnessContrast, \
#     HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
#     ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur

import cv2

# def get_transform(
#         target_size=256,
#         transform_list='horizontal_flip', # random_crop | keep_aspect
#         # augment_ratio=0.5,
#         is_train=True,
#         ):
#     transform = list()
#     transform_list = transform_list.split(', ')
#     # augments = list()

    
#     for transform_name in transform_list:
#         # default resize
#         transform.append(A.Resize(height=target_size, width=target_size,p=1))

#         if transform_name == 'random_crop':
#             # scale = (0.6, 1.0) if is_train else (0.8, 1.0)
#             transform.append(A.RandomResizedCrop(height=target_size, width=target_size,p=1))
#         # elif transform_name == 'resize':
#         #     transform.append(Resize(target_size))
#         elif transform_name == 'horizontal_flip':
#             transform.append(A.HorizontalFlip(p=0.5))
#         elif transform_name == 'vertical_flip':
#             transform.append(A.VerticalFlip(p=0.5))
#         elif transform_name == 'griddropout':
#             transform.append(A.GridDropout())


#     # transform.append(RandomApply(augments, p=augment_ratio))   
#     transform.append(ToTensorV2())
#     transform.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
#     return A.Compose(transform)

# def create_train_transforms(args):
    
#     return args.tr_aug

# def create_val_transforms(args):
    
#     return args.tr_aug

# def create_train_transforms(args, size=224):
#     if args.model=='vit_small_patch16_224':
#         return A.Compose([A.Resize(224, 224)])
#     else:
#         return A.Compose([
#             A.Resize(args.input_size, args.input_size), 
#             A.Normalize(),
#             # ToTensorV2()
#             ])

# def create_val_transforms(args, size=224):
#     if args.model=='vit_small_patch16_224':
#         return A.Compose([A.Resize(224, 224)])
#     else:
#         return A.Compose([
#             A.Resize(args.input_size, args.input_size), 
#             A.Normalize(),
#             # ToTensorV2()
#             ])
