import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    #"resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    #"resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet50": "./pretrained/resnet50-19c8e357.pth",
    "resnet101": "./pretrained/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, stride_at_1x1=False, dilation=1):
        super(Bottleneck, self).__init__()

        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride1x1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride3x3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, 4 * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if inplanes != planes * 4 or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes, 4 * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(4 * planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)

        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class ResNet(nn.Module):
    def __init__(self, block, block_num, stride=32):
        super(ResNet, self).__init__()
        assert stride in (8, 16, 32)
        dils = [1, 1] if stride == 32 else [el * (16 // stride) for el in (1, 2)]
        strds = [2 if el == 1 else 1 for el in dils]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.layer1 = self._make_layer(block, 64, block_num[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(
            block, 256, block_num[2], stride=strds[0], dilation=dils[0]
        )
        self.layer4 = self._make_layer(
            block, 512, block_num[3], stride=strds[1], dilation=dils[1]
        )

    def _make_layer(self, block, planes, b_num, stride=1, dilation=1):
        blocks = [block(self.inplanes, planes, stride=stride, dilation=dilation)]
        self.inplanes = planes * block.expansion
        for i in range(1, b_num):
            blocks.append(block(self.inplanes, planes, stride=1, dilation=dilation))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat4, feat8, feat16, feat32

    def init_weight(self, url):
        state_dict = modelzoo.load_url(url,'./pretrained/')
        self_state_dict = self.state_dict()
        for k, v in self_state_dict.items():
            if k in state_dict.keys():
                self_state_dict.update({k: state_dict[k]})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if "bn" in name or "downsample.1" in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params


def resnet50(pretrained=False, stride=8):
    model = ResNet(Bottleneck, [3, 4, 6, 3], stride=stride)
    if pretrained:
        model.init_weight(model_urls["resnet50"])
    return model


def resnet101(pretrained=False, stride=8):
    model = ResNet(Bottleneck, [3, 4, 23, 3], stride=stride)
    if pretrained:
        model.init_weight(model_urls["resnet101"])
    return model
