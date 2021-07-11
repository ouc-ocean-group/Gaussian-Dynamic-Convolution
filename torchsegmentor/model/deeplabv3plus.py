import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsegmentor.backbone import make_backbone
from torchsegmentor.module.aspp import ASPPModule
from torchsegmentor.module.gdconv import GFDConv


class Decoder(nn.Module):
    def __init__(self, n_classes, low_c=256):
        super(Decoder, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_c, 48, 1, bias=True),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(256, n_classes, kernel_size=1, bias=False)

        self.init_weight()

    def forward(self, feat_low, feat_aspp):
        h, w = feat_low.size()[2:]
        feat_low = self.conv_low(feat_low)
        feat_aspp_up = F.interpolate(
            feat_aspp, (h, w), mode="bilinear", align_corners=True
        )
        feat_cat = torch.cat((feat_low, feat_aspp_up), dim=1)
        feat_out = self.final_conv(feat_cat)
        logits = self.classifier(feat_out)
        return logits

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class DeeplabV3plus(nn.Module):
    def __init__(self, backbone, inplanes, n_class):
        super(DeeplabV3plus, self).__init__()

        self.backbone = backbone
        self.aspp = ASPPModule(inplanes, 256, with_gp=False)
        self.decoder = Decoder(n_class)

    def forward(self, x):
        low_feat, _, _, feat = self.backbone(x)
        feat_aspp = self.aspp(feat)
        logits = self.decoder(low_feat, feat_aspp)
        logits = F.interpolate(
            logits, x.size()[-2:], mode="bilinear", align_corners=True
        )
        return logits


def make_deeplabv3plus(cfg):
    backbone = make_backbone(cfg.backbone, cfg.out_stride, True)
    inplanes = {"resnet50": 2048, "resnet101": 2048}
    deeplabv3plus = DeeplabV3plus(backbone, inplanes[cfg.backbone], cfg.n_class)
    return deeplabv3plus
