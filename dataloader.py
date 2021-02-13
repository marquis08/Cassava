from conf import *

import os
import cv2
import copy
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.functional import img_to_tensor
from tqdm import tqdm


class LeafDataset(Dataset):
    def __init__(self, args, image_paths, labels=None, transforms=None, use_masking=False, is_test=False):
        self.args = args
        self.image_paths = image_paths
        self.use_masking = use_masking
        self.labels = labels
        self.images = []
        # self.default_transforms = default_transforms
        self.transforms = transforms
        self.is_test = is_test
        
        print('########################### dataset loader to memory for Faster Loading')
        for image_path in tqdm(self.image_paths):
            if args.grayscale:
                # print("THIS IS GRAY")
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = np.expand_dims(image, -1)
            else:
                
                image = cv2.imread(image_path)
                # print("THIS IS 3CH : {}".format(image.shape))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = Image.fromarray(image)

            # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            #if args.clahe:
            #    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
            #    image = clahe.apply(image)

            self.images.append(image)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # if get image one by one
        #image_path = self.image_paths[index]
        #image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.images[index]
        if self.transforms:
            image = self.transforms(image=image)['image']

        
        
        # if self.args.grayscale:
        #     image = img_to_tensor(image)
        # else:
        #     image = img_to_tensor(image, {"mean": [0.485, 0.456, 0.406],
        #                            "std": [0.229, 0.224, 0.225]})

        # image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = np.transpose(image, (2,0,1)).astype(np.float32)


        if self.is_test:
            return torch.tensor(image)
        else:
            label = self.labels[index]
            return torch.tensor(image), torch.tensor(label)#torch.tensor(img, dtype=torch.float32),\
                 #torch.tensor(self.labels[index], dtype=torch.long)
