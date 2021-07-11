from torchsegmentor.backbone.resnet import resnet50, resnet101


backbones = {"resnet50": resnet50, "resnet101": resnet101}


def make_backbone(name, stride, pretrained=True):
    assert name in backbones
    return backbones[name](pretrained=pretrained, stride=stride)
