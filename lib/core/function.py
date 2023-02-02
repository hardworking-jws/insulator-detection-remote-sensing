# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
import pandas as pd
from Tools.lib.utils.utils import AverageMeter
from Tools.lib.utils.utils import get_confusion_matrix
from Tools.lib.utils.utils import adjust_learning_rate
from Tools.lib.utils.utils import get_world_size, get_rank
from Tools.lib.core.criterion import CrossEntropy, OhemCrossEntropy, BinaryDiceLoss, DiceLoss
from Tools.lib.datasets.cityscapes import Cityscapes

# def weight_calculate(batch):
#     image_count = 127 * 127 * 8
#     batch_mask = batch.cpu()
#     image = np.array(batch_mask)
#     image_flatten = image.flatten()
#     pandas_image = pd.Series(image_flatten)
#     cout = pandas_image.value_counts()
#     cout_bg = cout[0]
#     cout_is = cout[1]
#     # frequency_bg = cout_bg / image_count
#     # frequency_is = cout_is / image_count
#     all_count = cout_is + cout_is
#     frequency_bg = cout_bg / all_count
#     frequency_is = cout_is / all_count
#     # 0.5 / frequnency
#     weight1_bg = 0.5 / frequency_bg
#     weight1_is = 0.5 / frequency_is
#     # modify cityspace
#     weight2_bg = 1 / np.log(1.02 + frequency_bg)
#     weight2_is = 1 / np.log(1.02 + frequency_is)
#     weight3_is = np.e ** (-frequency_is) / np.e ** (-(frequency_is + frequency_bg))
#     weight3_bg = np.e ** (-frequency_bg) / np.e ** (-(frequency_is + frequency_bg))
#     weight_list = torch.FloatTensor([weight2_bg, weight2_is]).cuda()
#     return weight_list
def weight_calculate(batch):
    image_count = 127 * 127 * 8
    batch_mask = batch.cpu()
    image = np.array(batch_mask)
    image_flatten = image.flatten()
    pandas_image = pd.Series(image_flatten)
    cout = pandas_image.value_counts()
    cout_bg = cout[0]
    cout_is = cout[1]
    frequency_bg = cout_bg / image_count
    frequency_is = cout_is / image_count
    # 0.5 / frequnency
    weight1_bg = 0.5 / frequency_bg
    weight1_is = 0.5 / frequency_is
    # modify cityspace
    weight2_bg = 1 / np.log(1.02 + frequency_bg)
    weight2_is = 1 / np.log(1.02 + frequency_is)
    weight3_is = np.e ** (-frequency_is) / np.e ** (-(frequency_is + frequency_bg))
    weight3_bg = np.e ** (-frequency_bg) / np.e ** (-(frequency_is + frequency_bg))
    weight_list = torch.FloatTensor([weight1_bg, weight1_is]).cuda()
    return weight_list



def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp

def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict, device):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()
    # rank = 0
    # world_size = 1

    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch
        size = labels.size()
        images = images.to(device)
        labels = labels.long().to(device)
        weight_list = weight_calculate(labels)
        # criterion = nn.CrossEntropyLoss(weight=weight_list, ignore_index=255)
        if config.LOSS.USE_OHEM:
            criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                         thres=config.LOSS.OHEMTHRES,
                                         min_kept=config.LOSS.OHEMKEEP,
                                         weight=weight_list)
        else:
            criterion = nn.CrossEntropyLoss(weight=weight_list, ignore_index=255)
        # losses, _ = model(images, labels)
        pred = model(images)
        masks_pred = F.upsample(input=pred, size=(
            size[-2], size[-1]), mode='bilinear')
        # l, masks_pred = model(images, labels)
        losses = criterion(masks_pred, labels)
        loss = losses.mean()
        reduced_loss = reduce_tensor(loss)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, print_loss)
            logging.info(msg)
            
            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict, device):
    
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    #混淆矩阵
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.no_grad():
        for _, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)
            weight_list = weight_calculate(label)
            # criterion = nn.CrossEntropyLoss(weight=weight_list, ignore_index=255)
            if config.LOSS.USE_OHEM:
                criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                             thres=config.LOSS.OHEMTHRES,
                                             min_kept=config.LOSS.OHEMKEEP,
                                             weight=weight_list)
            else:
                criterion = nn.CrossEntropyLoss(weight=weight_list, ignore_index=255)
            pred = model(image)
            pred = F.upsample(input=pred, size=(
                        size[-2], size[-1]), mode='bilinear')
            loss = criterion(pred, label)
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print_loss = ave_loss.average()/world_size

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, mean_IoU, IoU_array

def show_in_img(img_path, mask_path, final_output_dir):
    for x, y in zip(img_path, mask_path):
        # img_mask = Image.open(y)
        img_mask = cv2.imread(y, 0)
        img_mask = cv2.resize(img_mask, (196, 196))
        img_mask_array = np.array(img_mask)
        img_plt = cv2.imread(x[0])
        image1, contours_cv, hierarchy = cv2.findContours(img_mask_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # img_plt = cv2.drawContours(img_plt, contours_cv, -1, (0, 0, 255))  # 绘画出边缘
        # name_txt = x[0].split('/')[-1]
        name_txt = x[0].split('\\')[-1]
        name_txt_ture = name_txt.split('.')[0]
        for i in range(0, len(contours_cv)):
            x_right = np.max(contours_cv[i][:, :, 0])
            x_left = np.min(contours_cv[i][:, :, 0])
            y_right = np.max(contours_cv[i][:, :, 1])
            y_left = np.min(contours_cv[i][:, :, 1])
            area = (x_right - x_left) * (y_right - y_left)
            if area < 150:
                pass
            else:
                cv2.rectangle(img_plt, (x_left, y_left), (x_right, y_right), (0, 0, 255), 1)
                with open('%s.txt' % name_txt_ture, "a") as f:
                    f.write('insulator ')
                    f.write(str(x_left))
                    f.write(' ')
                    f.write(str(y_left))
                    f.write(' ')
                    f.write(str(x_right))
                    f.write(' ')
                    f.write(str(y_right))
                    f.write('\n')
            # cv2.rectangle(img_plt, (x_left, y_left), (x_right, y_right), (0, 255, 0), 1)
        cv2.imwrite(final_output_dir + '/' + x[0].split('\\')[-1], img_plt)

def testval(config, test_dataset, testloader, model, 
        sv_dir='output', sv_pred=True):
    model.eval()
    mask_path = []
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                # print(pred.cuda().data.cpu().numpy())
                test_dataset.save_pred(pred, sv_path, name)
                mask_path.append(sv_path + '/' + str(name[0]) + '.png')
            
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))
    #将mask图恢复高原图上去
    img_path = []
    root = config.DATASET.ROOT + config.DATASET.TEST_SET2
    root = 'F:\JWS\project_insulator\code\HRNet-Semantic-Segmentation-pytorch-v1.1/tools\data\list\cityscapes/test.lst'
    for i in open(root).readlines():
        i = i.splitlines()
        img_path.append(i)
    show_in_img(img_path, mask_path, sv_path)
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc
def test(config, test_dataset, testloader, model, 
        sv_dir='output', sv_pred=True):
    mask_path = []
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
                mask_path.append(sv_path + '/' + str(name[0]) + '.png')
    #将mask图恢复高原图上去
    img_path = []
    root = 'F:\JWS\project_insulator\code\HRNet-Semantic-Segmentation-pytorch-v1.1/tools\data\list\cityscapes/test.lst'
    for i in open(root).readlines():
        i = i.splitlines()
        img_path.append(i)
    show_in_img(img_path, mask_path, sv_path)
