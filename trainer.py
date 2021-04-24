from conf import *
from dataloader import *

import gc
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from glob import glob
from shutil import copytree, ignore_patterns
from datetime import datetime, timedelta
from distutils.dir_util import copy_tree

def train(args, trn_cfg):
    model = trn_cfg['model']
    criterion = trn_cfg['criterion']
    optimizer = trn_cfg['optimizer']
    scheduler = trn_cfg['scheduler']
    device = trn_cfg['device']
    fold_num = trn_cfg['fold_num']

    ### fp 16
    scaler = torch.cuda.amp.GradScaler()

    best_epoch = 0
    best_val_score = 0.0

    ########################## 시작하면 폴더생성
    ctime = datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
    if args.timm:
        model_name = args.timm_model
    else:
        model_name = args.model
   
    save_path = f'../WEIGHTS/cassava_models/{ctime}_{model_name}_fold{fold_num}'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # else:
    #     save_path += "_new"
    # closing path
    save_path += "/"
    src_dir = '../Cassava-Leaf-Disease-Classification/'
    # all src file to model_weight dir(copytree only works with non-exist dir)
    copytree(src_dir, save_path, ignore=ignore_patterns('__pycache__', '*.md'))
    print("Copying all file From {} To {}".format(src_dir, save_path))

    # Train the model
    for epoch in range(args.epochs):

        start_time = time.time()
        
        if args.multi_loss: # criterion = dict of losses
            trn_loss, loss_name = train_one_epoch_multiloss(args, model, criterion, trn_cfg['train_loader'], optimizer, scheduler, device, scaler, epoch)
            val_loss, val_acc, val_score, loss_name = validation_multiloss(args, trn_cfg, model, criterion, trn_cfg['valid_loader'], device, epoch)
        else:
            trn_loss = train_one_epoch(args, model, criterion, trn_cfg['train_loader'], optimizer, scheduler, device, scaler)
            val_loss, val_acc, val_score = validation(args, trn_cfg, model, criterion, trn_cfg['valid_loader'], device)

        elapsed = time.time() - start_time

        lr = [_['lr'] for _ in optimizer.param_groups]

        if args.multi_loss:
            content = f'Fold {fold_num}, Epoch {epoch}/{args.epochs}, lr: {lr[0]:.7f},tr loss: {trn_loss:.5f}, val loss: {val_loss:.5f},val_acc: {val_acc:.4f}, val_f1: {val_score:.4f},time: {elapsed:.0f}, loss: {loss_name}'
        else:
            content = f'Fold {fold_num}, Epoch {epoch}/{args.epochs}, lr: {lr[0]:.7f},tr loss: {trn_loss:.5f}, val loss: {val_loss:.5f},val_acc: {val_acc:.4f}, val_f1: {val_score:.4f},time: {elapsed:.0f}'
        print(content)
        with open(save_path + f'log_fold{fold_num}.txt', 'a') as appender:
            appender.write(content + '\n')

        # save model weight
        if val_score > best_val_score:
            best_val_score = val_score
            model_save_path = save_path + 'best_score_fold' + str(fold_num) + '_{}.pth'.format(str(epoch).zfill(3))#f'_{epoch}.pth'
            torch.save(model.state_dict(), model_save_path)
            ## Keep best 3 epoch
            for path in sorted(glob('{}best_score_fold{}_*.pth'.format(str('/'.join(model_save_path.split('/')[:-1])+"/"), fold_num)))[:-3]:
                os.remove(path)
#                print("remove weight at: {}".format(path))

        if args.scheduler == 'Plateau':
            scheduler.step(val_score)
        else:
            scheduler.step()



def train_one_epoch(args, model, criterion, train_loader, optimizer, scheduler, device, scaler):
    model.train()
    trn_loss = 0.0

    bar = tqdm(train_loader)
    for images, labels in bar:
        optimizer.zero_grad()
        print("#############################")
        print(type(images))
        print(type(labels))
        print(images)
        labels = labels.long()
        if device:
            images = images.to(device)
            labels = labels.to(device)

        with torch.set_grad_enabled(True): # Enable Grad
            if args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
    #           optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        trn_loss += loss.item()
        bar.set_description('loss : % 5f' % (loss.item()))
    epoch_train_loss = trn_loss / len(train_loader)

    return epoch_train_loss



def validation(args, trn_cfg, model, criterion, valid_loader, device):
    model.eval()
    val_loss = 0.0
    total_labels = []
    total_outputs = []

    bar = tqdm(valid_loader)
    with torch.no_grad():
        for images, labels in bar:
            labels = labels.long()
            total_labels.append(labels)

            if device:
                images = images.to(device)
                labels = labels.to(device)

            # outputs = torch.sigmoid(model(images)) # if binary
            metric = torch.argmax(model(images), dim=1).cpu().detach().numpy()
            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()
            total_outputs.append(metric)

            bar.set_description('loss : %.5f' % (loss.item()))

    epoch_val_loss = val_loss / len(valid_loader)

    total_labels = np.concatenate(total_labels).tolist()
    total_outputs = np.concatenate(total_outputs).tolist()
    # total_outputs = np.argmax(total_outputs, 1)

    acc = accuracy_score(total_labels, total_outputs)
    metrics = f1_score(total_labels, total_outputs, average='macro')

    return epoch_val_loss, acc, metrics

def train_one_epoch_multiloss(args, model, loss_dict, train_loader, optimizer, scheduler, device, scaler, epoch):
    if epoch > args.multi_loss_epoch_thr - 1 :
        # print("LOSS: {}".format(args.multi_loss_list[1]))
        criterion = loss_dict[args.multi_loss_list[1]] # if ['CE','Label'], then Label is chosen.
        loss_name = args.multi_loss_list[1]
    else:
        # print("LOSS: {}".format(args.multi_loss_list[0]))
        criterion = loss_dict[args.multi_loss_list[0]]
        loss_name = args.multi_loss_list[0]

    model.train()
    trn_loss = 0.0

    bar = tqdm(train_loader)
    for images, labels in bar:

        optimizer.zero_grad()
        labels = labels.long()
        if device:
            images = images.to(device)
            labels = labels.to(device)

        with torch.set_grad_enabled(True): # Enable Grad
            if args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
    #           optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        trn_loss += loss.item()

        bar.set_description('loss : % 5f' % (loss.item()))
    epoch_train_loss = trn_loss / len(train_loader)

    return epoch_train_loss, loss_name


def validation_multiloss(args, trn_cfg, model, loss_dict, valid_loader, device, epoch):
    
    if epoch > args.multi_loss_epoch_thr - 1 :
        # print("LOSS: {}".format(args.multi_loss_list[1]))
        criterion = loss_dict[args.multi_loss_list[1]] # if ['CE','Label'], then Label is chosen.
        loss_name = args.multi_loss_list[1]
    else:
        # print("LOSS: {}".format(args.multi_loss_list[0]))
        criterion = loss_dict[args.multi_loss_list[0]]
        loss_name = args.multi_loss_list[0]
        
    model.eval()
    val_loss = 0.0
    total_labels = []
    total_outputs = []

    bar = tqdm(valid_loader)
    with torch.no_grad():
        for images, labels in bar:
            labels = labels.long()
            total_labels.append(labels)

            if device:
                images = images.to(device)
                labels = labels.to(device)


            # outputs = torch.sigmoid(model(images)) # if binary
            metric = torch.argmax(model(images), dim=1).cpu().detach().numpy()
            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()
            total_outputs.append(metric)

            bar.set_description('loss : %.5f' % (loss.item()))

    epoch_val_loss = val_loss / len(valid_loader)

    total_labels = np.concatenate(total_labels).tolist()
    total_outputs = np.concatenate(total_outputs).tolist()
    # total_outputs = np.argmax(total_outputs, 1)

    acc = accuracy_score(total_labels, total_outputs)
    metrics = f1_score(total_labels, total_outputs, average='macro')

    return epoch_val_loss, acc, metrics, loss_name
