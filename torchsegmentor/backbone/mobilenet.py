from torch import nn
import torch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, BatchNorm2d, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            BatchNorm2d(out_planes)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, BatchNorm2d, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            BatchNorm2d(inp * expand_ratio),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=stride, padding=dilation,
                      dilation=dilation, groups=inp * expand_ratio, bias=False),
            BatchNorm2d(inp * expand_ratio),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            BatchNorm2d(oup, activation='none'),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, BatchNorm2d, width_mult=1.0, stride=8):
        super(MobileNetV2, self).__init__()
        self.num_features = 320
        scale = int(stride / 8)
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, int(scale), int(2 / scale)],
            [6, 96, 3, 1, int(2 / scale)],
            [6, 160, 3, 1, int(2 / scale)],
            [6, 320, 1, 1, int(2 / scale)],
        ]

        input_channel = int(32 * width_mult)
        self.features = [ConvBNReLU(3, input_channel, stride=2, BatchNorm2d=BatchNorm2d)]
        # building inverted residual blocks
        for t, c, n, s, dilate in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t, dilation=dilate, BatchNorm2d=BatchNorm2d))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t, dilation=dilate, BatchNorm2d=BatchNorm2d))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        xs = []
        for layer in self.features:
            x = layer(x)
            print(x.size())
            xs.append(x)
        return xs


def mobilenet_v2(BatchNorm2d, pretrained=False, stride=8, **kwargs):
    return MobileNetV2(BatchNorm2d=BatchNorm2d, stride=stride)


if __name__ == "__main__":
    net = mobilenet_v2()
    img = torch.randn(2, 3, 224, 224)
    net(img)