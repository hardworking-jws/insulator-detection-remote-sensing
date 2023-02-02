# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

# ------------------------------------------------
#VERSION = 'FPN_Res101_20181201'
VERSION = 'FPN_Res101_Tower_SV1/voc_6000model.ckpt'
VERSION_WV = 'FPN_Res101_Tower_WV/voc_9999model.ckpt'
NET_NAME = 'resnet_v1_101'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
### I:\backups\FPN_Tensorflow-master\tools\dist\
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0"
SHOW_TRAIN_INFO_INTE = 10
SMRY_ITER = 100
SAVE_WEIGHTS_INTE = 2000

SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/tools/inference_image'
INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_results'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise NotImplementedError

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights/')

EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/'
test_annotate_path = '/home/yjr/DataSet/VOC/VOC_test/VOC2007/Annotations'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
IS_FILTER_OUTSIDE_BOXES = False
FIXED_BLOCKS = 0  # allow 0~3
USE_07_METRIC = False
CUDA9 = True

RPN_LOCATION_LOSS_WEIGHT = 1.
RPN_CLASSIFICATION_LOSS_WEIGHT = 1.0

FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0
RPN_SIGMA = 3.0
FASTRCNN_SIGMA = 1.0

MUTILPY_BIAS_GRADIENT = None   # 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = None   # 10.0  if None, will not clip

EPSILON = 1e-5
MOMENTUM = 0.9
LR = 0.0003  # 0.001  # 0.0003
# DECAY_STEP = [60000, 80000]  # 50000, 70000
# MAX_ITERATION = 150000
#DECAY_STEP = [4000, 7000]  # 50000, 70000
DECAY_STEP = [2000, 4000]  # 50000, 70000
MAX_ITERATION = 6000

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'pascal'  # 'ship', 'spacenet', 'pascal', 'coco'
#PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN = [58.13, 85.38, 47.26]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
# IMG_SHORT_SIDE_LEN = 400  # 600  # 600
# IMG_MAX_LENGTH = 500  # 1000  # 1000
# IMG_SHORT_SIDE_LEN = 128  # 600  # 600
# IMG_MAX_LENGTH = 128# 1000  # 1000
IMG_SHORT_SIDE_LEN = 1024  # 600  # 600
IMG_MAX_LENGTH = 1024 # 1000  # 1000
# CLASS_NUM = 20
CLASS_NUM = 1

# --------------------------------------------- Network_config
BATCH_SIZE = 1
INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_DECAY = 0.00004 if NET_NAME.startswith('Mobilenet') else 0.0001

# ---------------------------------------------Anchor config
USE_CENTER_OFFSET = False

LEVLES = ['P2', 'P3', 'P4', 'P5', 'P6']
BASE_ANCHOR_SIZE_LIST = [30, 60, 80, 120, 160]  # 高景的tower
#BASE_ANCHOR_SIZE_LIST = [15, 20, 30, 50, 70]  # 小图的绝缘子
#BASE_ANCHOR_SIZE_LIST = [20, 30, 40, 60, 80]  # 大图的绝缘子
#BASE_ANCHOR_SIZE_LIST = [10, 18, 24, 32, 48]  # 高景图像的绝缘子
# ANCHOR_STRIDE_LIST = [4, 8, 16, 32, 64]
ANCHOR_STRIDE_LIST = [4, 8, 16, 32, 64]
#ANCHOR_STRIDE_LIST = [4, 4, 8, 16, 32]
ANCHOR_SCALES = [1.0]
# ANCHOR_RATIOS = [0.5, 1., 2.0]
ANCHOR_RATIOS = [0.5, 1., 2.0, 1/3., 3., 1.5, 1/1.5]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0]
ANCHOR_SCALE_FACTORS = None

# --------------------------------------------FPN config
SHARE_HEADS = True
KERNEL_SIZE = 3
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
TRAIN_RPN_CLOOBER_POSITIVES = False

RPN_MINIBATCH_SIZE = 256
RPN_POSITIVE_RATE = 0.5
RPN_NMS_IOU_THRESHOLD = 0.7
RPN_TOP_K_NMS_TRAIN = 12000
RPN_MAXIMUM_PROPOSAL_TARIN = 2000

RPN_TOP_K_NMS_TEST = 6000
RPN_MAXIMUM_PROPOSAL_TEST = 1000

# specific settings for FPN
# FPN_TOP_K_PER_LEVEL_TRAIN = 2000
# FPN_TOP_K_PER_LEVEL_TEST = 1000

# -------------------------------------------Fast-RCNN config
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 1.0
SHOW_SCORE_THRSHOLD = 0.6  # only show in tensorboard

FAST_RCNN_NMS_IOU_THRESHOLD = 0.3  # 0.6
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 100
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
FAST_RCNN_IOU_NEGATIVE_THRESHOLD = 0.0   # 0.1 < IOU < 0.5 is negative
FAST_RCNN_MINIBATCH_SIZE = 256  # if is -1, that is train with OHEM
FAST_RCNN_POSITIVE_RATE = 0.25

ADD_GTBOXES_TO_TRAIN = False


