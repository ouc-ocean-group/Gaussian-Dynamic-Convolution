import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ConvBlock
from .gdconv import GFDConv
from .gdconv import GSDConv
from .gdconv import GFDreConv

class ASPPModule(nn.Module):
    def __init__(self, input_channel, output_channel, with_gp=True):
        super(ASPPModule, self).__init__()
        self.with_gp = with_gp
        self.conv1 = ConvBlock(
            input_channel, output_channel, kernel_size=1, dilation=1, padding=0
        )
        '''self.conv2 = ConvBlock(
            input_channel, output_channel, kernel_size=3, dilation=6, padding=6
        )
        self.conv3 = ConvBlock(
            input_channel, output_channel, kernel_size=3, dilation=12, padding=12
        )'''
        #self.conv2 = GFDConv(input_channel,output_channel,scale=0.3,device="cuda")
        #self.conv3=GFDConv(input_channel,output_channel,scale=0.3,device="cuda")
        self.conv2 = GFDreConv(input_channel,output_channel,scale=1.0,device="cuda",base= 6)
        self.conv3= GFDreConv(input_channel,output_channel,scale=1.0,device="cuda",base= 15)
        #self.conv2 = GSDConv(input_channel,output_channel,2,4,6,scale=2)
        #self.conv3 = GSDConv(input_channel, output_channel, 2, 4, 12, scale=2)
        self.conv4 = ConvBlock(
            input_channel, output_channel, kernel_size=3, dilation=18, padding=18
        )
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBlock(input_channel, output_channel, kernel_size=1)
            self.conv_out = ConvBlock(output_channel * 5, output_channel, kernel_size=1)
            #self.conv_out = ConvBlock(output_channel * 4, output_channel, kernel_size=1)
        else:
            self.conv_out = ConvBlock(output_channel * 4, output_channel, kernel_size=1)
            #self.conv_out = ConvBlock(output_channel * 3, output_channel, kernel_size=1)


        self.init_weight()

    def forward(self, x):
        h, w = x.size()[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)

        if self.with_gp:
            avg = self.avg(x)
            feat5 = self.conv1x1(avg)
            feat5 = F.interpolate(feat5, (h, w), mode="bilinear", align_corners=True)
            feat = torch.cat((feat1, feat2,feat3, feat4, feat5), dim=1)
            #feat = torch.cat((feat1, feat2, feat3, feat5), dim=1)
        else:
            feat = torch.cat((feat1, feat2, feat3, feat4), dim=1)
            #feat = torch.cat((feat1, feat2, feat3), dim=1)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)
