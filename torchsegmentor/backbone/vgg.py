# This is the modified VGG16 by Deeplab-LargeFOV
# L.-C. Chen, G. Papandreou, I. Kokkinos, K. M. 0002, and A. L. Yuille, “Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs.,” ICLR, 2015.
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, in_dim=3, *args, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)
        layers = []
        layers.append(
            nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(3, stride=2, padding=1))

        layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(3, stride=2, padding=1))

        layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(3, stride=2, padding=1))

        layers.append(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(3, stride=1, padding=1))

        layers.append(nn.Conv2d(512, 512, kernel_size=3,
                                stride=1, padding=2, dilation=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(512, 512, kernel_size=3,
                                stride=1, padding=2, dilation=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(512, 512, kernel_size=3,
                                stride=1, padding=2, dilation=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(3, stride=1, padding=1))
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == "__main__":
    img = torch.randn(1, 3, 224, 224)
    net = VGG(stride=16)
    out = net(img)
    print(out.size())