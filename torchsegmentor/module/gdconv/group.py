import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import scipy.stats as stats
import numpy as np


class GroupGSDConv(nn.Module):
    def __init__(self, in_c, out_c, min_range, max_range, base_range=0, bias=False, scale=3):
        super(GroupGSDConv, self).__init__()
        self.base_range = base_range
        self.min_range = min_range
        self.max_range = max_range

        self.in_c = in_c
        self.out_c = out_c

        self.weight = Parameter(torch.randn(size=(out_c, in_c, 3, 3), dtype=torch.float))
        if bias:
            self.bias = Parameter(torch.randn((out_c,), dtype=torch.float))
        else:
            self.register_parameter('bias', None)

        self.dis = stats.truncnorm((min_range - base_range) / scale, (max_range - base_range) / scale, base_range, scale)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias.uniform_(-stdv, stdv)

    def forward(self, x):
        offset, unique_offset = self.sample_process()
        idx = torch.zeros_like(offset, dtype=torch.long)
        last_i = 0

        outs = []
        for o in unique_offset:
            offset_flag = offset == o
            w = self.weight[offset_flag].view(-1, self.in_c, 3, 3)
            b = self.bias[offset_flag] if self.bias is not None else None
            out = F.conv2d(x, w, dilation=int(o), padding=int(o), bias=b)
            outs.append(out)
            idx[offset_flag] = torch.arange(last_i, last_i+w.size(0), device=idx.device, dtype=torch.long)
        outs = torch.cat(outs, dim=1)
        outs = outs[:, idx, :, :]
        return outs

    def sample_process(self):
        offset = torch.as_tensor(self.dis.rvs(self.out_c), dtype=torch.float)
        offset = torch.round(offset)
        unique_offset = offset.unique()
        return offset, unique_offset


if __name__ == '__main__':
    conv = GroupGSDConv(128, 32, 2, 4, 3)
    img = torch.randn(1, 128, 32, 32)
    o = conv(img)
    print(o.shape)