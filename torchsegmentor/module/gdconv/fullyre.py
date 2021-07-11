import torch.nn as nn
import torch
import torch.nn.functional as F


class HalfNormal(object):
    def __init__(self, scale, seed, device):
        self.scale = scale
        self.device = device
        torch.manual_seed(seed)

    def sample(self, sample_shape=torch.Size()):
        result = torch.zeros(sample_shape).to(self.device)
        result = result.normal_(mean=0, std=self.scale).abs()
        return result


class GFDreConv(nn.Module):
    def __init__(self, in_features,out_features, bias=False, scale=0.1, device='cpu', seed=307, base=0):
        super(GFDreConv, self).__init__()
        self.conv = nn.Conv2d(9 * in_features, out_features, 1, bias=bias)
        self.dis = HalfNormal(scale, seed, device)
        self.size = None
        self.device = device
        self.base = base
        self.direction_basis = torch.tensor([[-1, 1, -1, 1, -1, 1, 0, 0],
                                             [-1, -1, 1, 1, 0, 0, -1, 1]]).float().view(-1).to(self.device)

    def forward(self, feat):
        sample_coor = self.sample_process(feat.size(2), feat.size(3))
        sample_coor_x = sample_coor[:, :, :8]
        sample_coor_y = sample_coor[:, :, 8:]
        feat = F.pad(feat, [1, 1, 1, 1])
        offset_feat = feat[:, :, sample_coor_y, sample_coor_x]
        offset_feat = F.pad(offset_feat.permute(0, 4, 1, 2, 3).contiguous()
                            .view(offset_feat.size(0), -1, offset_feat.size(2), offset_feat.size(3)), [1, 1, 1, 1])
        feat = torch.cat([feat, offset_feat], dim=1)
        feat = self.conv(feat)[:, :, 1:-1, 1:-1]
        return feat

    def sample_process(self, h, w):
        if self.size is None:
            if self.base == 0:
                size = torch.tensor([w] * 8 + [h] * 8).float().to(self.device)
            else :
                size = torch.tensor([self.base] * 8 + [self.base] * 8).float().to(self.device)
            yy = torch.linspace(0, h - 1, steps=h).unsqueeze(1).repeat(1, w).unsqueeze(-1).to(self.device)
            xx = torch.linspace(0, w - 1, steps=w).unsqueeze(0).repeat(h, 1).unsqueeze(-1).to(self.device)
            base_coor = torch.cat([xx.repeat(1, 1, 8), yy.repeat(1, 1, 8)], dim=-1).to(self.device)
        else:
            size = self.size
            base_coor = self.base_coor
        sample_ = self.dis.sample(torch.Size([h, w, 16]))
        offset = sample_ * self.direction_basis * size
        sample_coor = base_coor + offset
        sample_coor[:, :, :8] = torch.clamp(sample_coor[:, :, :8], min=0, max=w - 1)
        sample_coor[:, :, 8:] = torch.clamp(sample_coor[:, :, 8:], min=0, max=h - 1)
        return (sample_coor + 1).long()

