import torch
import torch.nn as nn
import torch.nn.functional as F


class PSPModule(nn.Module):
    def __init__(self, inplanes, pooling_size=[1, 2, 3, 6]):
        super(PSPModule, self).__init__()
        self.layers = len(pooling_size)
        self.pyramid_layers = self._make_pyramid_layers(inplanes, pooling_size)

    @staticmethod
    def _make_pyramid_layers(inplanes, pooling_size):
        pyramid_layers = []
        reduce_c = int(inplanes / len(pooling_size))
        for ps in pooling_size:
            pyramid_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(ps),
                    nn.Conv2d(inplanes, reduce_c, 1, bias=False),
                    nn.BatchNorm2d(reduce_c),
                )
            )
        return nn.ModuleList(pyramid_layers)

    def forward(self, x):
        input_size = x.size()
        ppm_out = [x]
        for layer in self.pyramid_layers:
            ppm_out.append(
                F.interpolate(
                    layer(x),
                    (input_size[2], input_size[3]),
                    mode="bilinear",
                    align_corners=True,
                )
            )

        ppm_out = torch.cat(ppm_out, 1)
        return ppm_out
