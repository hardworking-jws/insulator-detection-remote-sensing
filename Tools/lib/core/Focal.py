
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=5, size_average=True):

        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, (float, int)):    #仅仅设置第一类别的权重
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)
        if isinstance(alpha, list):  #全部权重自己设置
            self.alpha = torch.Tensor(alpha)
        self.gamma = gamma


    def forward(self, inputs, targets):
        alpha = self.alpha
        print('aaaaaaa',alpha)
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
        print('ppppppppppppppppppppp', P)
        # ---------one hot start--------------#
        class_mask = inputs.data.new(N, C).fill_(0)  # 生成和input一样shape的tensor
        print('依照input shape制作:class_mask\n', class_mask)
        class_mask = class_mask.requires_grad_()  # 需要更新， 所以加入梯度计算
        ids = targets.view(-1, 1)  # 取得目标的索引
        print('取得targets的索引\n', ids)
        class_mask.data.scatter_(1, ids.data, 1.)  # 利用scatter将索引丢给mask
        print('targets的one_hot形式\n', class_mask)  # one-hot target生成
        # ---------one hot end-------------------#
        probs = (P * class_mask).sum(1).view(-1, 1)
        print('留下targets的概率（1的部分），0的部分消除\n', probs)
        # 将softmax * one_hot 格式，0的部分被消除 留下1的概率， shape = (5, 1), 5就是每个target的概率

        log_p = probs.log()
        print('取得对数\n', log_p)
        # 取得对数
        loss = torch.pow((1 - probs), self.gamma) * log_p
        batch_loss = -alpha *loss.t()  # 對應下面公式
        print('每一个batch的loss\n', batch_loss)
        # batch_loss就是取每一个batch的loss值

        # 最终将每一个batch的loss加总后平均
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        print('loss值为\n', loss)
        return loss

class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        print("one_hot_key is \n", one_hot_key)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
torch.manual_seed(50) #随机种子确保每次input tensor值是一样的
input = torch.randn(5, 5, dtype=torch.float32, requires_grad=True)
print('input值为\n', input)
targets = torch.randint(5, (5, ))
print('targets值为\n', targets)
targets2 = torch.randn(5, 5, dtype=torch.float32, requires_grad=True)
print('targets2值为\n', targets2)

criterion = focal_loss()
loss = criterion(input, targets)
loss.backward()

criterion2 = MultiFocalLoss(num_class=1)
loss2 = criterion2(input, targets)
loss2.backward()
