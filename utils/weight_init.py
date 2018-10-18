import torch
import torch.nn as nn


def weight_init(w):
    if isinstance(w, nn.Conv2d):
        nn.init.kaiming_normal_(w.weight.data, a=0, mode='fan_in')
    elif isinstance(w, nn.BatchNorm2d):
        w.weight.data.fill_(1)
        w.bias.data.zero_()
