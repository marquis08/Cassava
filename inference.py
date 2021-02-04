#!/usr/bin/env python
# -*- coding: utf-8 -*-

from conf import *

import os
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from dataloader import *
from transforms import *
from models import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device", device)
test_df = pd.read_csv('/DATA/testset-for_user.csv', header=None)
test_image_paths = [os.path.join('/DATA', test_df[0][i], test_df[1][i]) for i in range(test_df.shape[0])]

if args.DEBUG:
    test_image_paths = test_image_paths[:500]

print(test_df.shape, len(test_image_paths))

bs = args.infer_batch_size
test_transforms = create_val_transforms(args, args.input_size)

print(test_transforms)
test_dataset = SleepDataset(args, 
                            image_paths=test_image_paths, 
                            labels=None,
                            transforms=test_transforms, 
                            is_test=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=bs, num_workers=8, shuffle=False, pin_memory=True)


# 테스트셋 예측 함수
def inference(model, test_loader, device):

    test_preds = np.zeros((len(test_loader.dataset), 5))
    bar = tqdm(test_loader)
    with torch.no_grad():
        for i, images in enumerate(bar):
            images = images.to(device)
            outputs = model(images)
            test_preds[i*bs:(i+1)*bs, :] = torch.sigmoid(outputs).detach().cpu().numpy()
    return test_preds

model = build_model(args, device) 
# 임의로 설정한 모델

model_path = '/USER/Sleep/models'
folder = args.infer_model_path#'2021-02-04_17:26:13_tf_efficientnet_b0_ns'
file_name = args.infer_best_model_name#'best_score_fold0_029.pth'
model.load_state_dict(torch.load(os.path.join(model_path, folder, file_name)))
model.eval()

test_preds = inference(model, test_loader, device)
np.save(os.path.join(model_path, folder, 'test_preds.npy'), test_preds)

test_preds = np.argmax(test_preds, 1)

result_df = pd.DataFrame(test_preds)
label_dict = {0:'Wake', 1:'N1', 2:'N2', 3:'N3', 4:'REM'}

result_df[0] = result_df[0].map(label_dict)
print(result_df.loc[:10], result_df.shape)

test_pred_path = "/USER/INFERENCE"
result_df.to_csv(os.path.join(test_pred_path, 'test_result.csv'), header=None, index=False)
# files.csv

