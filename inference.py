# -*- coding:utf-8 -*-
# author Jin Weishi -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
sys.path.append("../")

from Tools.lib.config import config
from Tools.lib.config import update_config
from Tools.lib.models.seg_hrnet import get_seg_model
from Tools.lib.datasets.cityscapes_cpu import Cityscapes
from Tools.lib.utils.utils import get_world_size
import torch.distributed as dist
from tqdm import tqdm
from torch.nn import functional as F
# sys.path.append("../")
import tensorflow as tf
import cv2
from osgeo import osr, gdal
# import gdal
import warnings
from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network,build_whole_network_WV
from libs.box_utils import draw_box_in_img
from help_utils import tools
from warnings import simplefilter
import numpy as np
import math
import torch
import torch.nn as nn
import shutil
from osgeo import gdal
import logging
from collections import OrderedDict
from torchvision.utils import make_grid
from PIL import Image
import argparse
import time
import pandas as pd
import requests
#忽略警告信息
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#0 -1 -2 only boxes    -1 -1 0 all boxes score and label
NOT_DRAW_BOXES = 0
ONLY_DRAW_BOXES = -1
ONLY_DRAW_BOXES_WITH_SCORES = -2
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

'''定义深度学习基础功能块'''
####################
# Useful tools
####################
def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer
def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer
def addnoise_cn(x, use_gpu):
    if use_gpu:
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    x = x.cpu()
    x = x.numpy()
    batchsize, d, m, n = x.shape
    x1 = np.zeros((batchsize, d + 1, m, n), dtype=x.dtype)
    x1[:, 0:d, :, :] = x
    for i in range(0, batchsize):
        x1[i, d, :, :] = np.random.random(size=(m, n))
    return torch.from_numpy(x1).to(device)
def addnoise(x, use_gpu):
    if use_gpu:
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    x = x.cpu()
    x = x.numpy()
    n = np.random.random(size=x.shape)
    x = x + 0.1 * n
    return torch.from_numpy(x).to(device)
def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer
def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding
class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr
class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr
def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)
def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0
    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
                  dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        if norm_type == 'weight':
            return c
        else:
            n = norm(norm_type, out_nc) if norm_type else None
            return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)
####################
# Useful blocks
####################
class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
                 bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
                           norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
                           norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res
class ResNetBlock_wn(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
                 bias=True, pad_type='zero', norm_type='weight', act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock_wn, self).__init__()

        act_type = None
        # norm_type = None
        # 定义基础卷积层0
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
                           norm_type, act_type, mode)
        # 定义基础卷积层1
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
                           norm_type, act_type, mode)
        # 定义激活层
        act = nn.ReLU(True)
        # 定义归一化函数
        wn = lambda x: torch.nn.utils.weight_norm(x)
        res = []
        res.append(
            wn(conv0)
        )
        res.append(act)
        res.append(wn(conv1))
        #        if mode == 'CNA':
        #            act_type = None
        #        if mode == 'CNAC':  # Residual path: |-CNAC-|
        #            act_type = None
        #            norm_type = None
        #        conv1 = conv_block(mid_nc*3, out_nc/2, kernel_size, stride, dilation, groups, bias, pad_type, \
        #            norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = nn.Sequential(*res)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res
####################
# Upsampler
####################
def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                       pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                      pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)
def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                 pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                      pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)

'''定义超分基础模型框架'''
'''SRGAN'''
class SRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='batch', act_type='relu', \
            mode='CAN', res_scale=1, upsample_mode='pixelshuffle'):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]
        LR_conv = conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = sequential(fea_conv, ShortcutBlock(sequential(*resnet_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x
'''SRGAN_Noise'''
class SRResNet_Noise(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='batch', act_type='relu', \
            mode='CAN', res_scale=1, upsample_mode='pixelshuffle', use_gpu=True):
        super(SRResNet_Noise, self).__init__()
        self.use_gpu = use_gpu
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = conv_block(in_nc+1, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]
        LR_conv = conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = sequential(fea_conv, ShortcutBlock(sequential(*resnet_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
#        noise = torch.randn(x.shape[1:4], device='cuda')
        x = addnoise_cn(x, self.use_gpu)
        x = self.model(x)
        return x
'''WDSR'''
class WDSRNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='weight', act_type='relu', \
            mode='CAN', res_scale=1, upsample_mode='pixelshuffle'):
        super(WDSRNet, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        head=[]
        body=[]
        tail=[]
        skip=[]
        head.append(
            wn(nn.Conv2d(in_nc, nf//2, 3, padding=3//2)))
        for _ in range(nb):
            body.append(ResNetBlock_wn(nf//2, nf*3, nf//2, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale))
        out_feats = upscale*upscale*out_nc
        tail.append(
                wn(conv_block(nf//2, out_feats, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=norm_type, act_type=None))
                )
        tail.append(nn.PixelShuffle(upscale))
        skip.append(
                wn(conv_block(in_nc, out_feats, kernel_size=5, stride=1, bias=True, \
                        pad_type='zero', norm_type=norm_type, act_type=None))
                )
        skip.append(nn.PixelShuffle(upscale))
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        s=self.skip(x)
        x=self.head(x)
        x=self.body(x)
        x=self.tail(x)
        x += s
        return x


'''定义基础超分框架'''
class BaseModel():
    def __init__(self, load_path, modeltype, use_gpu):
        self.use_gpu=use_gpu
        if self.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
#        self.is_train = opt['is_train']
#        self.device = torch.device('cpu')
        self.schedulers = []
        self.optimizers = []
        self.load_path=load_path

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_step):
        save_filename = '{}_{}.pth'.format(iter_step, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
'''定义超分模型'''
logger = logging.getLogger('base')
class Model_Builder(BaseModel):
    '''构建网络模型加载框架'''
    def __init__(self, load_path, modeltype, use_gpu):
        super(Model_Builder, self).__init__(load_path, modeltype, use_gpu)
        # define network and load pretrained models
        if modeltype=='WDSR':
            self.netG = WDSRNet(in_nc=4, out_nc=4, nf=64, nb=23, upscale=4, norm_type='weight', act_type='relu', mode='CNA', upsample_mode='pixelshuffle')
        elif modeltype=='SRGAN':
            self.netG = SRResNet(in_nc=4, out_nc=4, nf=64, nb=16, upscale=4, norm_type=None, act_type='relu', mode= 'CNA', upsample_mode='pixelshuffle')
        elif modeltype=='SRGAN_Noise':
            self.netG = SRResNet_Noise(in_nc=4, out_nc=4, nf=64, nb=16, upscale=4, norm_type= None, act_type='relu', mode='CNA', upsample_mode='pixelshuffle', use_gpu=self.use_gpu)
        else:
            raise NotImplementedError('Model [{:s}] not recognized.'.format(modeltype))
        self.netG=self.netG.to(self.device)
        self.load()
        self.print_network()

    def feed_data(self, data, need_HR=False):
        self.var_L = data.to(self.device)  # LR
        if need_HR:
            self.real_H = data['HR'].to(self.device)  # HR

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()
        for k, v in self.netG.named_parameters():
            v.requires_grad = False

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)

        for k, v in self.netG.named_parameters():
            v.requires_grad = True
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=False):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):
        load_path_G = self.load_path
        if load_path_G is not None:
            self.load_network(load_path_G, self.netG)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)

'''读取卫星影像函数'''
def read_img(filename):
    '''读取带坐标的Tif文件'''
    dataset=gdal.Open(filename)       #打开文件

    im_width = dataset.RasterXSize    #栅格矩阵的列数
    im_height = dataset.RasterYSize   #栅格矩阵的行数

    im_geotrans = list(dataset.GetGeoTransform())  #仿射矩阵
    im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵

    del dataset #清理内存
    return im_proj,im_geotrans,im_data

'''存储卫星影像函数'''
def write_img(filename,im_proj,im_geotrans,im_data):
    '''保存tif文件'''
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

        #判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

        #创建文件
    driver = gdal.GetDriverByName("GTiff")            #数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)              #写入仿射变换参数
    dataset.SetProjection(im_proj)                    #写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  #写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset

'''tensor转遥感影像函数'''
def tensor2rsimg(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
    elif n_dim == 3:
        img_np = tensor.numpy()
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

'''影像拉伸函数'''
def imgstretch(img):
    gray=img
    d2=np.percentile( gray,2 )
    #d2=np.min(gray)
    u98=np.percentile( gray,98 )
    #u98=np.max(gray)
    maxout=255
    minout=0
    gray_new=minout + ( (gray-d2) / (u98-d2) ) * (maxout - minout)
    gray_new[gray_new < minout]=minout
    gray_new[gray_new > maxout]=maxout
    gray_out=Image.fromarray(gray_new.astype(np.uint8))
    return gray_out

'''清空缓存文件夹'''
def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

'''进度条函数'''
# def process_bar(percent, start_str='', end_str='', total_length=0):
#     bar = ''.join(["\033[31m%s\033[0m"%'   '] * int(percent * total_length)) + ''
#     bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str
#     print(bar, end='', flush=True)
def process_bar(percent, start_str='', end_str='', total_length=0):
    a='*'* int(100 * percent)
    b='.'* (100 - int(100 * percent))
    c=int(100 * percent)
    print("{:^3.0f}%[{}->{}]".format(c, a, b))
    time.sleep(0.1)

'''直接超分辨率函数'''
def SuperResolution(im_data, ModType, SaveForDetect=True, SateType='SV1', use_gpu = True):
    '''
    超分辨率函数
    '''
    ################################################
    im_data = im_data.astype(np.float32)/255.
    im_data = np.expand_dims(im_data, axis=0)
    im_data = torch.tensor(im_data)
    ################################################读取tif文件并转为tensor格式
    LoadPath = './models/SRmodels/'+SateType+'/'+ModType+'.pth'
    model = Model_Builder(LoadPath, ModType, use_gpu)
    #print('Model created')
    ################################################创建超分辨率模型
    m ,n = im_data.shape[2], im_data.shape[3]
    m1 = math.ceil(m/600)
    n1 = math.ceil(n/600)
    b = torch.zeros(1,4,m1*600,n1*600)
    b[:,:,0:m,0:n]=im_data
    del im_data
    b_SR = np.zeros((4,m1*2400,n1*2400), dtype=np.uint8)
    count=100000
    for i in range(0, m1):
        for j in range (0, n1):
            #影像超分辨率
            count=count+1
            if torch.sum(b[:,:,i*600:i*600+600,j*600:j*600+600],(1,2,3))>0:
                model.feed_data(b[:,:,i*600:i*600+600,j*600:j*600+600])
                model.test()
                visuals = model.get_current_visuals(need_HR=False)
                sr_img = tensor2rsimg(visuals['SR'])
                b_SR[:,i*2400:i*2400+2400,j*2400:j*2400+2400]=sr_img
    del b
    if SaveForDetect:
        # 判断是否采用伪彩色融合
        b_SR1=np.zeros((3,m*4,n*4), dtype=np.uint8)
        # b_SR1[0,:,:]=b_SR[2,:m*4,:n*4]
        # b_SR1[1,:,:]=b_SR[3,:m*4,:n*4]
        # b_SR1[2,:,:]=b_SR[0,:m*4,:n*4]
        #
        b_SR1[0,:,:]=b_SR[2,:m*4,:n*4]
        b_SR1[1,:,:]=b_SR[3,:m*4,:n*4]
        b_SR1[2,:,:]=b_SR[0,:m*4,:n*4]
    else:
        b_SR1=np.zeros((4,m*4,n*4), dtype=np.uint8)
        b_SR1[0,:,:]=b_SR[0,:m*4,:n*4]
        b_SR1[1,:,:]=b_SR[1,:m*4,:n*4]
        b_SR1[2,:,:]=b_SR[2,:m*4,:n*4]
        b_SR1[3,:,:]=b_SR[3,:m*4,:n*4]
    del b_SR
    return b_SR1

'''超分辨率总体函数'''
def DoSR(InPath, ModType, OutPath, Stretch, SaveForDetect, SateType, DoWhole, use_gpu):
    im_proj, im_geotrans, data = read_img(InPath)
    step = 0
    end_str = '100%'
    # 判断是否需要灰度拉伸，若是则执行
    if Stretch=='True':
        print("灰度拉伸中")
        process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
        data[0, :, :] = imgstretch(data[0, :, :])
        step = step + 25
        process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
        data[1, :, :] = imgstretch(data[1, :, :])
        step = step + 25
        process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
        data[2, :, :] = imgstretch(data[2, :, :])
        step = step + 25
        process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
        data[3, :, :] = imgstretch(data[3, :, :])
        step = step + 25
        process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
        print("灰度拉伸完成")
        step = 0
    # 执行超分辨率部分
    if DoWhole=='True':
        # 首先判断是否针对整景影像进行超分，若是则执行过程如下
        m, n = data.shape[1], data.shape[2]
        m1 = math.ceil(m / 1200)
        n1 = math.ceil(n / 1200)
        b = np.zeros((4, 1200 * m1, 1200 * n1), dtype=np.uint8)
        b[:, 0:m, 0:n] = data
        del data
        # 影像边长补齐为1200的整数倍
        count = 100000
        if SaveForDetect=='True':
            # 判断是否使用假彩色
            print("影像超分辨率缓存中")
            process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
            b_SR = np.zeros((3, 4800, 4800), dtype=np.uint8)
            progress = 0
            for i in range(0, m1):
                for j in range(0, n1):
                    # 执行超分辨率并缓存
                    b_SR = SuperResolution(b[:, i * 1200:i * 1200 + 1200, j * 1200:j * 1200 + 1200],
                                           ModType, SaveForDetect, SateType, use_gpu)
                    geotrans = np.array(im_geotrans)
                    xoffset2 = j * 1200
                    yoffset2 = i * 1200
                    px = geotrans[0] + xoffset2 * geotrans[1] + yoffset2 * geotrans[2]
                    py = geotrans[3] + xoffset2 * geotrans[4] + yoffset2 * geotrans[5]
                    geotrans[0] = px
                    geotrans[3] = py
                    geotrans[1] /= 4
                    geotrans[5] /= 4
                    if geotrans[5] > 0:
                        geotrans[5] = -geotrans[5]
                    geotrans = geotrans.tolist()
                    i1 = count + i
                    j1 = count + j
                    progress += 1
                    write_img('./cache/' + str(i1) + str(j1) + '.tif', im_proj, geotrans, b_SR)
                    step = 100*progress/(m1*n1)
                    process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
            del b
            del b_SR
            print("缓存完成")
            # 超分辨率结果拼接
            step = 0
            print("超分辨率结果拼接中")
            output = np.zeros((3, 4 * m, 4 * n), dtype=np.uint8)
            progress = 0
            for i in range(0, m1):
                for j in range(0, n1):
                    if i == m1 - 1:
                        mi = 4 * m
                    else:
                        mi = i * 4800 + 4800
                    if j == n1 - 1:
                        nj = 4 * n
                    else:
                        nj = j * 4800 + 4800
                    im_proj2, im_geotrans2, im_data = read_img(
                        './cache/' + str(count + i) + str(count + j) + '.tif')
                    output[:, i * 4800:mi, j * 4800:nj] = im_data[:, :mi - i * 4800, :nj - j * 4800]
                    progress += 1
                    step = 100 * progress / (m1 * n1)
                    process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
            # 超分辨率结果存储
            print("拼接完成")
            step = 0
            print("超分辨率影像存储中")
            geotrans2 = np.array(im_geotrans)
            geotrans2[1] /= 4
            geotrans2[5] /= 4
            if geotrans2[5] > 0:
                geotrans2[5] = -geotrans2[5]
            write_img(OutPath, im_proj, geotrans2.tolist(), output)
            del output
            print("存储完成")
        else:
            # 不采用假彩色合成的超分辨率过程
            print("影像超分辨率缓存中")
            b_SR = np.zeros((4, 4800, 4800), dtype=np.uint8)
            progress = 0
            for i in range(0, m1):
                for j in range(0, n1):
                    b_SR = SuperResolution(b[:, i * 1200:i * 1200 + 1200, j * 1200:j * 1200 + 1200],
                                           ModType, SaveForDetect, SateType, use_gpu)
                    geotrans = np.array(im_geotrans)
                    xoffset2 = j * 1200
                    yoffset2 = i * 1200
                    px = geotrans[0] + xoffset2 * geotrans[1] + yoffset2 * geotrans[2]
                    py = geotrans[3] + xoffset2 * geotrans[4] + yoffset2 * geotrans[5]
                    geotrans[0] = px
                    geotrans[3] = py
                    geotrans[1] /= 4
                    geotrans[5] /= 4
                    if geotrans[5] > 0:
                        geotrans[5] = -geotrans[5]
                    geotrans = geotrans.tolist()
                    i1 = count + i
                    j1 = count + j
                    write_img('./cache/' + str(i1) + str(j1) + '.tif', im_proj, geotrans, b_SR)
                    progress += 1
                    step = 100 * progress / (m1 * n1)
                    process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
            del b
            del b_SR
            print("缓存完成")
            step = 0
            print("超分辨率结果拼接中")
            output = np.zeros((4, 4 * m, 4 * n), dtype=np.uint8)
            progress = 0
            for i in range(0, m1):
                for j in range(0, n1):
                    if i == m1 - 1:
                        mi = 4 * m
                    else:
                        mi = i * 4800 + 4800
                    if j == n1 - 1:
                        nj = 4 * n
                    else:
                        nj = j * 4800 + 4800
                    im_proj2, im_geotrans2, im_data = read_img(
                        './cache/' + str(count + i) + str(count + j) + '.tif')
                    output[:, i * 4800:mi, j * 4800:nj] = im_data[:, :mi - i * 4800, :nj - j * 4800]
                    progress += 1
                    step = 100 * progress / (m1 * n1)
                    process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
            print("拼接完成")
            print("超分辨率影像存储中")
            geotrans2 = np.array(im_geotrans)
            geotrans2[1] /= 4
            geotrans2[5] /= 4
            if geotrans2[5] > 0:
                geotrans2[5] = -geotrans2[5]
            write_img(OutPath, im_proj, geotrans2.tolist(), output)
            del output
            print("存储完成")
        del_file('./cache/')
    else:
        # 针对裁剪后影像的超分辨率过程（省略缓存步骤）
        print("影像超分辨率中")
        data = data.astype(np.float32) / 255.
        data = np.expand_dims(data, axis=0)
        data = torch.tensor(data)
        ####################################################图片转为tensor格式
        LoadPath = './models/SRmodels/' + SateType + '/' + ModType + '.pth'
        model = Model_Builder(LoadPath, ModType, use_gpu)
        ####################################################创建超分辨率模型
        m, n = data.shape[2], data.shape[3]
        m1 = math.ceil(m / 600)
        n1 = math.ceil(n / 600)
        b = torch.zeros(1, 4, m1 * 600, n1 * 600)
        b[:, :, 0:m, 0:n] = data
        del data
        b_SR = np.zeros((4, m1 * 2400, n1 * 2400), dtype=np.uint8)
        count = 100000
        progress = 0
        for i in range(0, m1):
            for j in range(0, n1):
                count = count + 1
                if torch.sum(b[:, :, i * 600:i * 600 + 600, j * 600:j * 600 + 600], (1, 2, 3)) > 0:
                    model.feed_data(b[:, :, i * 600:i * 600 + 600, j * 600:j * 600 + 600])
                    model.test()
                    visuals = model.get_current_visuals(need_HR=False)
                    sr_img = tensor2rsimg(visuals['SR'])
                    b_SR[:, i * 2400:i * 2400 + 2400, j * 2400:j * 2400 + 2400] = sr_img
                    progress += 1
                    step = 100 * progress / (m1 * n1)
                    process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
        del b
        #####################################################影像超分辨率
        print("超分辨率完成")
        step = 0
        print("影像拼接中")
        if SaveForDetect=='True':
            b_SR1 = np.zeros((3, m * 4, n * 4), dtype=np.uint8)
            b_SR1[0, :, :] = b_SR[2, :m * 4, :n * 4]
            step += 1
            process_bar(step / 3, start_str='', end_str=end_str, total_length=15)
            b_SR1[1, :, :] = b_SR[3, :m * 4, :n * 4]
            step += 1
            process_bar(step / 3, start_str='', end_str=end_str, total_length=15)
            b_SR1[2, :, :] = b_SR[0, :m * 4, :n * 4]
            step += 1
            process_bar(step / 3, start_str='', end_str=end_str, total_length=15)
        else:
            b_SR1 = np.zeros((4, m * 4, n * 4), dtype=np.uint8)
            b_SR1[0, :, :] = b_SR[0, :m * 4, :n * 4]
            step += 25
            process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
            b_SR1[1, :, :] = b_SR[1, :m * 4, :n * 4]
            step += 25
            process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
            b_SR1[2, :, :] = b_SR[2, :m * 4, :n * 4]
            step += 25
            process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
            b_SR1[3, :, :] = b_SR[3, :m * 4, :n * 4]
            step += 25
            process_bar(step / 100, start_str='', end_str=end_str, total_length=15)
        del b_SR
        print("拼接完成")
        print("超分辨率影像存储中")
        geotrans2 = np.array(im_geotrans)
        geotrans2[1] /= 4
        geotrans2[5] /= 4
        if geotrans2[5] > 0:
            geotrans2[5] = -geotrans2[5]
        write_img(OutPath, im_proj, geotrans2.tolist(), b_SR1)
        del b_SR1
        print("存储完成")

'''定义杆塔识别模块'''
def read_img_dt(filename):
    '''读取带坐标的Tif文件'''
    dataset = gdal.Open(filename)       #打开文件

    im_width = dataset.RasterXSize    #栅格矩阵的列数
    im_height = dataset.RasterYSize   #栅格矩阵的行数

    im_geotrans = list(dataset.GetGeoTransform())  #仿射矩阵
    im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵

    del dataset #清理内存
    return im_data

def write_img_dt(filename, im_data):
    '''保存tif文件'''
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

        #判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    #创建文件
    driver = gdal.GetDriverByName("GTiff")            #数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    #dataset.SetGeoTransform(im_geotrans)              #写入仿射变换参数
    #dataset.SetProjection(im_proj)                    #写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  #写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset

def get_file_names(data_dir, file_type=['tif', 'tiff']):
    '''裁剪及合并过程中读取影像文件名'''
    result_dir = []
    result_name = []
    for maindir, subdir, file_name_list in os.walk(data_dir):
        for filename in file_name_list:
            apath = maindir + '/' + filename
            ext = apath.split('.')[-1]
            if ext in file_type:
                result_dir.append(apath)
                result_name.append(filename)
            else:
                pass
    return result_dir, result_name


def get_same_img(img_dir, img_name):
    result = {}
    for idx, name in enumerate(img_name):
        temp_name = ''
        for idx2, item in enumerate(name.split('_')[:-4]):
            if idx2 == 0:
                temp_name = temp_name + item
            else:
                temp_name = temp_name + '_' + item

        if temp_name in result:
            result[temp_name].append(img_dir[idx])
        else:
            result[temp_name] = []
            result[temp_name].append(img_dir[idx])
    return result


def assign_spatial_reference_byfile(src_path, dst_path):
    '''
    融合坐标信息
    :param src_path: 原始整景影像路径
           dst_path：需要融合坐标信息的影像路径
    :return: None
    '''
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    sr = osr.SpatialReference()
    sr.ImportFromWkt(src_ds.GetProjectionRef())
    geoTransform = src_ds.GetGeoTransform()
    dst_ds = gdal.Open(dst_path, gdal.GA_Update)
    dst_ds.SetProjection(sr.ExportToWkt())
    dst_ds.SetGeoTransform(geoTransform)
    #print(geoTransform)
    dst_ds = None
    src_ds = None

def assign_spatial_reference_byfile_coordinate(src_path):
    '''
    获取坐标信息
    :param src_path: 原始整景影像路径
    :return: 坐标信息
    '''
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    sr = osr.SpatialReference()
    sr.ImportFromWkt(src_ds.GetProjectionRef())
    geoTransform = src_ds.GetGeoTransform()
    return geoTransform


def cut(in_dir, out_dir, file_type=['tif', 'tiff'], out_type='tif', out_size=1024):
    '''
    裁剪整景影像
    :param in_dir: 原始整景影像
           out_dir：裁剪后存放的文件夹路径
           out_size: 这里默认为1024
    :return: None
    '''
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data_dir_list, _ = get_file_names(in_dir, file_type)
    count = 0
    print('Cut begining for ', str(len(data_dir_list)), ' images.....')
    for each_dir in data_dir_list:
        time_start = time.time()
        # image = np.array(io.imread(each_dir))
        #image = np.array(Image.open(each_dir))
        image = np.array(read_img_dt(each_dir))
        image = np.transpose(image, (1,2,0))
        #影像整体分割、分割因素是cut_factor_row x cut_factor_clo
        cut_factor_row = int(np.ceil(image.shape[0] / out_size))
        cut_factor_clo = int(np.ceil(image.shape[1] / out_size))
        for i in range(cut_factor_row):
            for j in range(cut_factor_clo):

                if i == cut_factor_row - 1:
                    i = image.shape[0] / out_size - 1
                else:
                    pass

                    if j == cut_factor_clo - 1:
                        j = image.shape[1] / out_size - 1
                    else:
                        pass

                start_x = int(np.rint(i * out_size))
                start_y = int(np.rint(j * out_size))
                end_x = int(np.rint((i + 1) * out_size))
                end_y = int(np.rint((j + 1) * out_size))

                temp_image = image[start_x:end_x, start_y:end_y, :]

                out_dir_images = out_dir + '/' + each_dir.split('/')[-1].split('.')[0] \
                                 + '_' + str(start_x) + '_' + str(end_x) + '_' + str(start_y) + '_' + str(end_y) + '.' + out_type
                #out_dir_images = out_dir + '/' + str(i) + '_' + str(j) + '.' + out_type
                #out_dir_images = out_dir + '/' + '_' + str(start_x) + '_' + str(end_x) + '_' + str(start_y) + '_' + str(end_y) + '.' + out_type

                out_image = Image.fromarray(temp_image)
                out_image.save(out_dir_images)

                src_path = 'I:/project_insulator/code/FPN_Tensorflow-master/tools/Tif_Cut/6.tif'  # 带地理影像
                assign_spatial_reference_byfile(args.OutPath, out_dir_images)

        count += 1
        print('End of ' + str(count) + '/' + str(len(data_dir_list)) + '...')
        time_end = time.time()
        print('Time cost: ', time_end - time_start)
    print('Cut Finsh!')
    return 0

def combine(data_dir, w, h, c, out_dir, out_type='tif', file_type=['tif', 'tiff']):
    '''
    合并裁剪识别后的影像
    :param data_dir: 识别后的裁剪影像
           w：整景影像的宽
           h：整景影像的高
           c: 通道数，默认为3，RGB
           out_dir：合并的文件路径
    :return: None
    '''
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img_dir, img_name = get_file_names(data_dir, file_type)
    print('\n')
    print('Combine begining for ', str(len(img_dir)), ' images.....')
    dir_dict = get_same_img(img_dir, img_name)
    count = 0
    for key in dir_dict.keys():
        temp_label = np.zeros(shape=(w, h, c), dtype=np.uint8)
        dir_list = dir_dict[key]
        for item in dir_list:
            name_split = item.split('_')
            x_start = int(name_split[-4])
            x_end = int(name_split[-3])
            y_start = int(name_split[-2])
            y_end = int(name_split[-1].split('.')[0])
            img = Image.open(item)
            img = np.array(img)
            temp_label[x_start:x_end, y_start:y_end, :] = img

        img_name = key + '.' + out_type
        new_out_dir = out_dir + '/' + img_name
        temp_label = np.transpose(temp_label, (2, 0, 1))
        write_img_dt(new_out_dir, temp_label)
        #由于inference之后影像丧失了坐标信息，这里通过与原始影像进行地理坐标融合
        assign_spatial_reference_byfile(args.OutPath, new_out_dir)
        count += 1
        print('End of ' + str(count) + '/' + str(len(dir_dict)) + '...')
    print('Combine Finsh!')
    return 0

#从空间坐标系转换到地理坐标系得到杆塔坐标范围
def Get_coordinate(geoTransform, box, x_cut, y_cut):
    '''
    获得地理坐标
    :param geoTransform: GDAL地理数据
           box：输出的识别框空间分辨率坐标
           x_cut：分割影像的x坐标增加量
           y_cut：分割影像的y坐标增加量
    :return: 地理坐标
    '''
    column_left = box[0] + y_cut
    row_left = box[1] + x_cut
    column_right = box[2] + y_cut
    row_right = box[3] + x_cut
    Xmap_left = geoTransform[0] + column_left * geoTransform[1] + row_left * geoTransform[2]
    Ymap_left = geoTransform[3] + column_left * geoTransform[4] + row_left * geoTransform[5]
    Xmap_right = geoTransform[0] + column_right * geoTransform[1] + row_right * geoTransform[2]
    Ymap_right = geoTransform[3] + column_right * geoTransform[4] + row_right * geoTransform[5]
    return Xmap_left, Ymap_left, Xmap_right, Ymap_right

def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs

def geo2lonlat(src_path, x_left, y_left, x_right, y_right):
    '''
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    '''
    dataset = gdal.Open(src_path)
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords_left = ct.TransformPoint(x_left, y_left)
    coords_right = ct.TransformPoint(x_right, y_right)
    return coords_left[:2], coords_right[:2]
'''定义识别杆塔号模块'''
def Haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）

    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    # c = 2 * math.asin(math.sqrt(a))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000
def get_tower_label(excel_path, lon, lat):
    df = pd.read_excel(excel_path)
    dis = df.apply(lambda x: Haversine(x['经度'], x['纬度'], lon, lat), axis=1)
    index_get = dis.nsmallest(1).index
    return index_get[0], df['杆塔编号'][index_get[0]], df['线路'][index_get[0]]
'''定义绝缘子串识别模块'''
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

def test(config, test_dataset, testloader, model,
         sv_dir='output', sv_pred=True):
    args = parse_args()
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
                name_null = []
                name = name[0]
                name_null.append(name)
                sv_path = sv_dir
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name_null)
                mask_path.append(sv_path + '/' + str(name[0])[2:] + '.png')
                del name_null[:]


def insulator_detect():
    args = parse_args()
    # logger.info(pprint.pformat(args))
    # logger.info(pprint.pformat(config))

    # cudnn related setting
    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # cudnn.deterministic = config.CUDNN.DETERMINISTIC
    # cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = get_seg_model(config)
    if args.SateType == 'WV':
        model_state_file = './models/Insulatormodels/best_WV.pth'
    elif args.SateType == 'SV1':
        model_state_file = './models/Insulatormodels/best_SV1.pth'
    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    pretrained_dict = torch.load(model_state_file, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # gpus = list(config.GPUS)
    # model = nn.DataParallel(model, device_ids=gpus).cuda()
    # model = nn.DataParallel(model, device_ids=gpus)

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = Cityscapes(
        root=config.DATASET.ROOT,
        test_img_files='./Tower/',
        if_test=True,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size= test_size,
        downsample_rate=1)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    # start = timeit.default_timer()
    test(config,
         test_dataset,
         testloader,
         model,
         sv_dir='./insulator')

    # end = timeit.default_timer()
    # logger.info('Mins: %d' % np.int((end - start) / 60))

def show_in_img(x, y, final_output_dir, tower_area):
    img_mask = cv2.imread(y, 0)
    img_mask_array = np.array(img_mask)
    img_plt = cv2.imread(x)
    contours_cv, hierarchy = cv2.findContours(img_mask_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    name_txt = x.split('\\')[-1]
    name_txt_ture = name_txt.split('.')[0]
    for i in range(0, len(contours_cv)):
        x_right = np.max(contours_cv[i][:, :, 0]) + int(tower_area[0][0])
        x_left = np.min(contours_cv[i][:, :, 0]) + int(tower_area[0][0])
        y_right = np.max(contours_cv[i][:, :, 1]) + int(tower_area[0][1])
        y_left = np.min(contours_cv[i][:, :, 1]) + int(tower_area[0][1])
        area = (x_right - x_left) * (y_right - y_left)
        if area < 200:
            pass
        elif area >4500:
            pass
        else:
            cv2.rectangle(img_plt, (x_left, y_left), (x_right, y_right), (0, 0, 255), 1)
    cv2.imwrite(x, img_plt)

def show_in_img_multi(x, y, final_output_dir, tower_area):
    img_mask = cv2.imread(y, 0)
    img_mask_array = np.array(img_mask)
    img_plt = cv2.imread(x)
    contours_cv, hierarchy = cv2.findContours(img_mask_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    name_txt = x.split('\\')[-1]
    name_txt_ture = name_txt.split('.')[0]
    for i in range(0, len(contours_cv)):
        x_right = np.max(contours_cv[i][:, :, 0]) + int(tower_area[0])
        x_left = np.min(contours_cv[i][:, :, 0]) + int(tower_area[0])
        y_right = np.max(contours_cv[i][:, :, 1]) + int(tower_area[1])
        y_left = np.min(contours_cv[i][:, :, 1]) + int(tower_area[1])
        area = (x_right - x_left) * (y_right - y_left)
        if area < 200:
            pass
        elif area >4500:
            pass
        else:
            cv2.rectangle(img_plt, (x_left, y_left), (x_right, y_right), (0, 0, 255), 1)
    cv2.imwrite(x, img_plt)

def detect(det_net, inference_save_path, real_test_imgname_list):
    '''检测过程'''
    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not GBR
    img_batch = tf.cast(img_plac, tf.float32)
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = tf.expand_dims(img_batch, axis=0)# [1, None, None, 3]

    detection_boxes, detection_scores, detection_category = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #定义存放坐标信息的空数组
    Map_write = []
    Geo_write = []
    box_write = []
    Geo_write_show = []
    do_insulator_list = '500'
    excel_path = './test.xls'
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        for i, a_img_name in enumerate(real_test_imgname_list):

            raw_img = cv2.imread(a_img_name)
            one_data0 = np.unique(raw_img[:, :, 0])
            one_data1 = np.unique(raw_img[:, :, 1])
            one_data2 = np.unique(raw_img[:, :, 2])
            if one_data0.any() == [0] and one_data1.any() == [0] and one_data2.any() == [0]:
                # start = time.time()
                nake_name = a_img_name.split('/')[-1]
                cv2.imwrite(inference_save_path + '/' + nake_name,
                            raw_img)
                # end = time.time()
                # tools.view_bar('{} image cost {}s'.format(a_img_name, (end - start)), i + 1,
                #                len(real_test_imgname_list))
            else:
                start = time.time()
                resized_img, detected_boxes, detected_scores, detected_categories = \
                    sess.run(
                        [img_batch, detection_boxes, detection_scores, detection_category],
                        feed_dict={img_plac: raw_img[:, :, ::-1]}  # cv is BGR. But need RGB
                    )
                end = time.time()

                #show_indices = detected_scores >= cfgs.SHOW_SCORE_THRSHOLD
                if args.SateType == 'SV1':
                    show_indices = detected_scores >= args.Threshold_SV1
                elif args.SateType == 'WV':
                    show_indices = detected_scores >= args.Threshold_WV
                show_scores = detected_scores[show_indices]
                show_boxes = detected_boxes[show_indices]
                show_categories = detected_categories[show_indices]

                labes = np.ones(shape=[len(show_boxes), ], dtype=np.float32) * ONLY_DRAW_BOXES
                scores = np.zeros_like(labes)
                # 为了防止检测到的杆塔范围没有把杆塔囊括完，将检测框略微扩大
                if args.SateType == 'SV1':
                    for i in range(0, len(show_boxes)):
                        get_area_weight = show_boxes[i][2] - show_boxes[i][0]
                        get_area_height = show_boxes[i][3] - show_boxes[i][1]
                        if get_area_weight < 196:
                            fit1 = (196 - get_area_weight) / 2
                            show_boxes[i][0] = show_boxes[i][0] - fit1
                            show_boxes[i][2] = show_boxes[i][2] + fit1
                            if show_boxes[i][0] < 0:
                                show_boxes[i][0] = 0
                            if show_boxes[i][2] < 0:
                                show_boxes[i][2] = 0
                        if get_area_height < 196:
                            fit2 = (196 - get_area_height) / 2
                            show_boxes[i][1] = show_boxes[i][1] - fit2
                            show_boxes[i][3] = show_boxes[i][3] + fit2
                            if show_boxes[i][1] < 0:
                                show_boxes[i][1] = 0
                            if show_boxes[i][3] < 0:
                                show_boxes[i][3] = 0
                        img_get = raw_img
                        img_box = img_get[int(show_boxes[i][1]):int(show_boxes[i][3]),
                                  int(show_boxes[i][0]):int(show_boxes[i][2]), :]
                        cv2.imwrite('./Tower' + '/' + '%d_' % i + a_img_name.split('/')[-1],
                                    img_box)
                elif args.SateType == 'WV':
                    for i in range(0, len(show_boxes)):
                        get_area_weight = show_boxes[i][2] - show_boxes[i][0]
                        get_area_height = show_boxes[i][3] - show_boxes[i][1]
                        if get_area_weight < 300:
                            fit1 = (300 - get_area_weight) / 2
                            show_boxes[i][0] = show_boxes[i][0] - fit1
                            show_boxes[i][2] = show_boxes[i][2] + fit1
                            if show_boxes[i][0] < 0:
                                show_boxes[i][0] = 0
                            if show_boxes[i][2] < 0:
                                show_boxes[i][2] = 0
                        if get_area_height < 300:
                            fit2 = (300 - get_area_height) / 2
                            show_boxes[i][1] = show_boxes[i][1] - fit2
                            show_boxes[i][3] = show_boxes[i][3] + fit2
                            if show_boxes[i][1] < 0:
                                show_boxes[i][1] = 0
                            if show_boxes[i][3] < 0:
                                show_boxes[i][3] = 0
                        img_get = raw_img
                        img_box = img_get[int(show_boxes[i][1]):int(show_boxes[i][3]),
                                  int(show_boxes[i][0]):int(show_boxes[i][2]), :]
                        cv2.imwrite('./Tower' + '/' + '%d_' % i + a_img_name.split('/')[-1],
                                    img_box)
                #获取对应的坐标信息
                nake_name = a_img_name.split('/')[-1]
                out_dir_images = inference_save_path + '/' + nake_name
                mask_name = a_img_name.split('/')[-1]
                mask_name = mask_name.split('.')[0]
                Map_initial = assign_spatial_reference_byfile_coordinate(args.OutPath)
                name_split = nake_name.split('_')
                x_start = int(name_split[-4])
                y_start = int(name_split[-2])
                show_write = []
                del show_write[:]
                del Geo_write_show[:]
                for box in show_boxes:
                    Map = Get_coordinate(Map_initial, box, x_start, y_start)
                    Geo = geo2lonlat(args.OutPath, Map[0], Map[1], Map[2], Map[3])
                    Map_write.append([nake_name, Map])
                    Geo_write.append([nake_name, Geo])
                    Geo_write_show.append([nake_name, Geo])
                box_write.append([nake_name, show_boxes])
                for one in Geo_write_show:
                    if one[1]:
                        lon = (one[1][0][1] + one[1][1][1]) / 2
                        lat = (one[1][0][0] + one[1][1][0]) / 2
                        _, tower_label_show, line_show = get_tower_label('./Tower_line.xls', lon, lat)
                        show_write.append([line_show, tower_label_show])
                # final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(np.squeeze(resized_img, 0),
                #                                                                     boxes=show_boxes,
                #                                                                     labels=show_categories,
                #                                                                     scores=show_scores,
                #                                                                     tower_labels=show_write)
                final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(np.squeeze(resized_img, 0),
                                                                                    boxes=show_boxes,
                                                                                    labels=show_categories,
                                                                                    scores=show_scores)

                cv2.imwrite(inference_save_path + '/' + nake_name,
                           final_detections[:, :, ::-1])
                insulator_dir = './insulator/'
                Tower_dir = './Tower/'
                ### 产生问题原因为，mask图被覆盖了
                if args.DOInsulator == 'True':
                    if show_write:
                        if do_insulator_list in show_write[0][0]:
                            # print('ssssssssssssssssssss')
                            if len(show_boxes):
                                if len(show_boxes) >1:
                                    for i in range(0, len(show_boxes)):
                                        insulator_detect()
                                        show_in_img_multi(args.save_cutdetect_dir + '/' +  a_img_name.split('/')[-1], './insulator/' + '%d_' % i + mask_name + '.png', args.save_cutdetect_dir, show_boxes[i])

                                        # show_in_img_multi('./Test_result/' + a_img_name.split('/')[-1], './insulator/' + '%d_' % i + mask_name + '.png', './Test_result', show_boxes[i])
                                    del_file(insulator_dir)
                                    # del_file(Tower_dir)
                                else:
                                    insulator_detect()
                                    show_in_img(args.save_cutdetect_dir + '/' + a_img_name.split('/')[-1], './insulator/' + '0_' + mask_name + '.png',
                                                args.save_cutdetect_dir, show_boxes)
                                    # show_in_img('./Test_result/' + a_img_name.split('/')[-1], './insulator/' + '0_' + mask_name + '.png',
                                    #             './Test_result', show_boxes)
                                    del_file(insulator_dir)
                                    # del_file(Tower_dir)
                #分割后检测的影像会丢失地理坐标信息，这里和原图进行地理坐标信息的融合，检测后的影像也会带有地理坐标信息了
                out_dir_images = inference_save_path + '/' + nake_name
                assign_spatial_reference_byfile(args.OutPath, out_dir_images)
                Map_initial = assign_spatial_reference_byfile_coordinate(args.OutPath)
                tools.view_bar('{} image cost {}s'.format(a_img_name, (end - start)), (i + 1), len(real_test_imgname_list))
                ########################################
                #把坐标结果写到txt中
        with open(args.txt_save_dir + "/" + "Map_Coordinate.txt", "w") as f:
            for i in Map_write:
                f.write(str(i[0]))
                f.write('    ')
                if i[1]:
                    f.write(str(i[1]))
                    f.write('\n')
                else:
                    f.write("No tower")
                    f.write('\n')
        f.close()
        with open(args.txt_save_dir + "/" + "Geo_Coordinate.txt", "w") as f:
            for i in Geo_write:
                f.write(str(i[0]))
                f.write('    ')
                if i[1]:
                    lon = (i[1][0][1] + i[1][1][1]) / 2
                    lat = (i[1][0][0] + i[1][1][0]) / 2
                    _, tower_label, line = get_tower_label('./Tower_line.xls', lon, lat)
                    f.write(str(i[1]))
                    f.write('   ')
                    f.write(str(line) + str(tower_label) + '号杆塔')
                    f.write('\n')
                else:
                    f.write("No tower")
                    f.write('\n')
        f.close()
        with open(args.txt_save_dir + "/" + "box.txt", "w") as f:
            for i in box_write:
                f.write(str(i[0]))
                f.write('    ')
                if i[1].any():
                    f.write(str(i[1]))
                    f.write('\n')
                else:
                    f.write("No tower")
                    f.write('\n')
        f.close()


def inference(test_dir, inference_save_path):

    test_imgname_list = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir)
                                                          if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    assert len(test_imgname_list) != 0, 'test_dir has no imgs there.' \
                                        ' Note that, we only support img format of (.jpg, .png, and .tiff) '
    if args.SateType == 'WV':
        faster_rcnn = build_whole_network_WV.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                           is_training=False)
    elif args.SateType == 'SV1':
        faster_rcnn = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                           is_training=False)
    detect(det_net=faster_rcnn, inference_save_path=inference_save_path, real_test_imgname_list=test_imgname_list)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='TestImgs...U need provide the test dir')
    parser.add_argument('InPath', type=str, help='影像输入路径')
    parser.add_argument('OutPath', type=str, help='输出路径')
    parser.add_argument('SateType', type=str, help='卫星类型，可选值为: SV1, GF2, GF1，WV。注：检测杆塔一定选择SV1或者WV模型，由于卫星分辨率目前只支持SV1、WV检测')
    parser.add_argument('ModType', type=str, help='模型类型，可选值为: WDSR, SRGAN, SRGAN_Noise')
    parser.add_argument('Stretch', type=str, help='是否灰度拉伸：True->拉伸, False->不拉伸。注：检测杆塔一定要拉伸')
    parser.add_argument('FakeColor', type=str, help='是否假彩色输出：True->假彩色输出, False->4通道输出。注：检测杆塔一定要假彩色输出')
    parser.add_argument('DoWhole', type=str, help='是否处理整景影像：'
                                                   'True->整景处理(适用于大幅影像), '
                                                   'False->处理部分影像(速度更快)')
    parser.add_argument('UseGPU', type=str, help='是否采用GPU处理：True->GPU处理, False->CPU处理')
    parser.add_argument('DoDetection', type=str, help='是否检测杆塔：True->检测杆塔, False->只进行超分')
    parser.add_argument('DOInsulator', type=str, help='是否检测绝缘子串：True->检测绝缘子串, False->只进行超分和杆塔识别')
    parser.add_argument('Savecutdetect', type=str, help='是否保留中间结果：True->保留, False->不保留')
    parser.add_argument('--Threshold_SV1', type=float, default=0.92, help='检测高景杆塔的阈值，可以根据检测结果自己调节，默认0.92')
    parser.add_argument('--Threshold_WV', type=float, default=0.96, help='检测Worldview杆塔的阈值，可以根据检测结果自己调节，默认0.96')
    parser.add_argument('--cut_data_dir', dest='cut_data_dir',
                        help='超分过后影像存放的文件夹，用作检测杆塔的输入',
                        default='./save', type=str)
    parser.add_argument('--save_cutdetect_dir', dest='save_cutdetect_dir',
                        help='影像裁剪识别输出路径',
                        default='./Test_result', type=str)
    parser.add_argument('--combine_save_data_dir', dest='combine_save_data_dir',
                        help='检测结果存放的文件夹',
                        default='./combine', type=str)
    parser.add_argument('--txt_save_dir', dest='txt_save_dir',
                        help='坐标输出txt存放的文件夹，默认在程序目录的Txt,注：必须输入，因为只进行超分或超分和识别都需要输出',
                        default='./Txt', type=str)
    parser.add_argument('--localhost', dest='localhost',
                        help='回调地址，用于接收程序运行失败或者成功信息的地址',
                        default='http://localhost:8091/inspection?taskId=1234', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    update_config(config, args)

    return args


if __name__ == '__main__':
    # #继承args输入参数
    args = parse_args()
    #超分部分
    # try:
    if args.UseGPU=='True':
        use_gpu = True
        GPU = '0'
    else:
        use_gpu = False
        GPU = '-1'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    if args.DoDetection == 'True':
        DoSR(args.InPath, args.ModType, args.OutPath, args.Stretch, args.FakeColor, args.SateType, args.DoWhole, use_gpu)
        #获取图片大小
        data_size = read_img_dt(args.OutPath)
        #裁剪图片大小
        cut_save_data_dir = './Out'
        data_dir = './Out/'
        # test_results_save_dir = './Test_result'
        if not os.path.exists(cut_save_data_dir):
            os.mkdir(cut_save_data_dir)
        if not os.path.exists(args.save_cutdetect_dir):
            os.mkdir(args.save_cutdetect_dir)
        cut(args.cut_data_dir, cut_save_data_dir, file_type=['tif', 'tiff'], out_type='tif', out_size=1024)
        inference(data_dir,
                  inference_save_path=args.save_cutdetect_dir)
        combine(args.save_cutdetect_dir, w=data_size.shape[1], h=data_size.shape[2], c=3, out_dir=args.combine_save_data_dir, out_type='tif', file_type=['tif'])
        #检测过后，清除裁剪的图片
        for root, dirs, files in os.walk(cut_save_data_dir):
            for name in files:
                if name.endswith(".tif"):
                    os.remove(os.path.join(root, name))
        if args.Savecutdetect == 'False':
            del_file(args.save_cutdetect_dir)
        with open(args.txt_save_dir + "/" + "Finished.txt", "w") as f:
            f.write("Finished!")
    else:
        DoSR(args.InPath, args.ModType, args.OutPath, args.Stretch, args.FakeColor, args.SateType, args.DoWhole,
             use_gpu)
        with open(args.txt_save_dir + "/" + "Finished.txt", "w") as f:
            f.write("Finished!")
    #     try:
    #         print('Program Successed，the success message is being passed to the callback address')
    #         requests.get(args.localhost + '&status=success')
    #         print('callback send successfully!')
    #     except:
    #         print('Error in callback address passing')
    # except:
    #     try:
    #         print('Program error，the failure message is being passed to the callback address')
    #         requests.get(args.localhost + '&status=fail')
    #         print('callback send successfully!')
    #     except:
    #         print('Error in callback address passing')