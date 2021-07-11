import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsegmentor.module.nn import ConvBlock


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(inplanes, planes, 3, stride=stride, bias=True)
        self.conv2 = ConvBlock(planes, planes, 3, 1, bias=True, relu=False)
        self.skip_conv = (
            ConvBlock(inplanes, planes, 1, stride=stride, bias=False, relu=False)
            if stride != 1 or inplanes != planes
            else None
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.skip_conv is not None:
            x = self.skip_conv(x)
        out = out + x
        return out


class ResidualStage(nn.Module):
    def __init__(self, inplanes, planes, num, stride=2):
        super(ResidualStage, self).__init__()
        layers = [ResidualBlock(inplanes, planes, stride)]
        for i in range(num - 1):
            layers.append(ResidualBlock(planes, planes, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PSPPath(nn.Module):
    def __init__(self, inplanes, planes, strides):
        super(PSPPath, self).__init__()
        self.conv = ConvBlock(inplanes, planes, 1, bias=False)
        self.scale = strides
        self.h = 60
        self.w = 34

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, self.scale)
        x = self.conv(x)
        x = F.interpolate(x, size=(self.h, self.w), mode="bilinear", align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, inplanes, planes):
        super(PSPModule, self).__init__()
        self.path32 = PSPPath(inplanes, planes, 32)
        self.path16 = PSPPath(inplanes, planes, 16)
        self.path8 = PSPPath(inplanes, planes, 8)
        self.path4 = PSPPath(inplanes, planes, 4)

    def forward(self, x):
        x32 = self.path32(x)
        x16 = self.path16(x)
        x8 = self.path8(x)
        x4 = self.path4(x)

        x = torch.cat((x32, x16, x8, x4), dim=1)
        return x


class DFNet(nn.Module):
    def __init__(self, layer_nums=(3, 3, 3, 1)):
        super(DFNet, self).__init__()
        self.conv_stage0 = ConvBlock(3, 32, 3, 2, bias=False)
        self.conv_stage1 = ConvBlock(32, 64, 3, 2, bias=False)
        self.conv_stage2 = ResidualStage(64, 64, layer_nums[0])
        self.conv_stage3 = ResidualStage(64, 128, layer_nums[1])
        self.conv_stage4 = ResidualStage(128, 256, layer_nums[2])
        self.conv_stage5 = ResidualStage(256, 512, layer_nums[3], stride=1)

        self.psp = PSPModule(512, 128)

        self.psp_conv = ConvBlock(512, 512, 3)

        self.decoder5 = nn.Sequential(
            ConvBlock(512, 128, 1),
            ConvBlock(128, 32, 1),
            nn.ConvTranspose2d(32, 32, 4, 2, padding=1, groups=32, bias=False),
        )

        self.skip4 = ConvBlock(128, 32, 1, bias=False)
        self.decoder4 = nn.Sequential(
            ConvBlock(64, 32, 3),
            ConvBlock(32, 19, 1),
            nn.ConvTranspose2d(19, 19, 4, 2, padding=1, groups=19, bias=False),
        )

        self.skip3 = ConvBlock(64, 19, 1, bias=False)
        self.decoder3 = nn.Sequential(
            ConvBlock(38, 19, 3),
            ConvBlock(19, 19, 3),
            nn.ConvTranspose2d(19, 19, 16, 8, padding=4, groups=19, bias=False),
        )

    def forward(self, x):
        x0 = self.conv_stage0(x)
        x1 = self.conv_stage1(x0)
        x2 = self.conv_stage2(x1)
        x3 = self.conv_stage3(x2)
        x4 = self.conv_stage4(x3)
        x5 = self.conv_stage5(x4)

        x5 = self.psp(x5)
        x5 = self.psp_conv(x5)
        d_x5 = self.decoder5(x5)
        s4 = self.skip4(x3)

        d_x5 = torch.cat((d_x5, s4), dim=1)
        d_x4 = self.decoder4(d_x5)
        s3 = self.skip3(x2)
        d_x4 = F.interpolate(
            d_x4, size=s3.size()[-2:], mode="bilinear", align_corners=True
        )

        d_x4 = torch.cat((d_x4, s3), dim=1)
        out = self.decoder3(d_x4)
        return out
