import torch.nn as nn
import torch.nn.functional as F

from torchsegmentor.backbone import make_backbone
from torchsegmentor.module.psp import PSPModule


class PSPNet(nn.Module):
    def __init__(self, backbone, inplanes, n_class, pooling_size=[1, 2, 3, 6]):
        super(PSPNet, self).__init__()

        self.backbone = backbone
        self.psp = PSPModule(inplanes, pooling_size)
        self.final_conv = nn.Sequential(
            nn.Conv2d(2 * inplanes, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, n_class, 1),
        )
        self.aux_conv = nn.Conv2d(int(inplanes / 2), n_class, 1)

    def forward(self, x):
        _, _, low_feat, feat = self.backbone(x)
        low_out = self.aux_conv(low_feat)

        feat = self.psp(feat)
        out = self.final_conv(feat)

        low_out = F.interpolate(
            low_out, size=x.size()[-2:], mode="bilinear", align_corners=True
        )
        out = F.interpolate(
            out, size=x.size()[-2:], mode="bilinear", align_corners=True
        )

        return out, low_out


def make_pspnet(cfg):
    backbone = make_backbone(cfg.backbone, cfg.out_stride, True)
    inplanes = {"resnet50": 1024, "resnet101": 2048}
    pspnet = PSPNet(backbone, inplanes[cfg.backbone], cfg.n_class)
    return pspnet
