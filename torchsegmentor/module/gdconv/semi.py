import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import scipy.stats as stats
import numpy as np


class GSDConv(nn.Module):
    def __init__(self, in_c, out_c, min_range, max_range, base_range=0, bias=False, scale=3, fix_eval=False):
        super(GSDConv, self).__init__()
        self.base_range = base_range
        self.min_range = min_range
        self.max_range = max_range

        self.fix_eval = fix_eval

        self.in_c = in_c
        self.out_c = out_c

        self.weight = Parameter(torch.FloatTensor(out_c, in_c, 3, 3))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_c))
        else:
            self.register_parameter('bias', None)

        self.dis = stats.truncnorm((min_range - base_range) / scale, (max_range - base_range) / scale, base_range, scale)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feat):
        offset = self.sample_process()
        feats = F.conv2d(feat, self.weight, dilation=offset, padding=offset, bias=self.bias)
        return feats

    def sample_process(self):
        if self.fix_eval and not self.training:
            return self.base_range
        offset = np.around(self.dis.rvs())
        return int(offset)

