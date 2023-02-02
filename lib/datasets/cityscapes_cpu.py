# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

#from .base_dataset import BaseDataset
from Tools.lib.datasets.base_dataset import BaseDataset
class Cityscapes(BaseDataset):
    def __init__(self, 
                 root, 
                 # list_path,
                 test_img_files,
                 if_test = True,
                 num_samples=None, 
                 num_classes=2,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=128,
                 crop_size=(128, 128),
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=16,
                 # mean=[0.485, 0.456, 0.406],
                 # std=[0.229, 0.224, 0.225]):
                 # mean = [0.232, 0.260, 0.169],
                 # std = [0.172, 0.202, 0.153]):
                 mean = [0.223, 0.240, 0.160],
                 std = [0.177, 0.201, 0.154]):

        super(Cityscapes, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.root = root
        # self.list_path = list_path
        self.test_img_files = test_img_files
        self.if_test = if_test
        self.num_classes = num_classes
        # self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
        #                                 1.0166, 0.9969, 0.9754, 1.0489,
        #                                 0.8786, 1.0023, 0.9539, 0.9843,
        #                                 1.1116, 0.9037, 1.0865, 1.0955,
        #                                 1.0865, 1.1529, 1.0507]).cuda()
        # one is 0.2 //22.7273 and 0.5118_miou=0.466  23.8711 and 1.4456 miou=0.496
        # self.class_weights = torch.FloatTensor([1.012, 2.012]).cuda()
        # self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
        #                                 1.0166, 0.9969, 0.9754, 1.0489,
        #                                 0.8786, 1.0023, 0.9539, 0.9843,
        #                                 1.1116, 0.9037, 1.0865, 1.0955,
        #                                 1.0865, 1.1529, 1.0507]).cuda()
        self.multi_scale = multi_scale
        self.flip = flip
        self.center_crop_test = center_crop_test
        
        # self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]
        self.label_mapping = {0: 0, 1: 1}



    def read_files(self):
        files = []
        if self.if_test:
            for root_dir, dirs, files_get in os.walk(self.test_img_files):
                for file in files_get:
                    img_file = os.path.join(str(root_dir), str(file))
                    name = os.path.splitext(os.path.basename(img_file))[0]
                    files.append({
                        "img": img_file,
                        "name": name,
                    })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        # image = cv2.imread(os.path.join(self.root,'cityscapes',item["img"]),
        #                    cv2.IMREAD_COLOR)
        image = cv2.imread(item['img'],
                           cv2.IMREAD_COLOR)
        size = image.shape

        if self.if_test:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(os.path.join(self.root,'cityscapes',item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        image, label = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, 
                                self.center_crop_test)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        # final_pred = torch.zeros([1, self.num_classes,
        #                             ori_height,ori_width]).cuda()

        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width])
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                # preds = torch.zeros([1, self.num_classes,
                #                            new_h,new_w]).cuda()
                # count = torch.zeros([1, 1, new_h, new_w]).cuda()

                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w])
                count = torch.zeros([1, 1, new_h, new_w])

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
            preds = F.upsample(preds, (ori_height, ori_width), 
                                   mode='bilinear')
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        # for image_item in self.img_list:
        #     image_path, label_path = image_item
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

    def show_in_img(self, img_path, mask_path, final_output_dir):
        for x, y in zip(img_path, mask_path):
            img_mask = Image.open(y)
            img_mask_array = np.array(img_mask)
            img_plt = cv2.imread(x)
            image1, contours_cv, hierarchy = cv2.findContours(img_mask_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img_plt = cv2.drawContours(img_plt, contours_cv, -1, (0, 0, 255))  # 绘画出边缘
            for i in range(0, len(contours_cv)):
                x_right = np.max(contours_cv[i][:, :, 0])
                x_left = np.min(contours_cv[i][:, :, 0])
                y_right = np.max(contours_cv[i][:, :, 1])
                y_left = np.min(contours_cv[i][:, :, 1])
                cv2.rectangle(img_plt, (x_left, y_left), (x_right, y_right), (0, 255, 0), 1)
            cv2.imwrite(final_output_dir + '/' + img_path.split('/')[-1], img_plt)

    def get_img_path(self):
        img_path = []
        for item in self.img_list:
            image_path, label_path = item
            img_path.append(image_path)
        return img_path


