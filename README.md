![](./assets/imgs/torch-segmentor-logo.png)

Pytorch models and toolbox for Semantic Segmentation.

## Requirements

- PyTorch >= 1.2.0
- easydict
- cv2
- skimage
- scipy
- tensorboardX
- ninja
- coco python api (if use COCO dataset)

## Features

### Dataset

We build many dataloaders for the popular segmentation datasets, including:
- Cityscapes
- ADE20K
- Pascal Context
- COCO

### Backbone

We build 5 wellknown backbone networks, i.e., DenseNet, MobileNet, ResNet, VGG, and Xception to extract feature maps.

### Module

We implement many famous and useful modules for boosting semantic segmentation performance:
- Syncronized Batch Normalization
- Atrous Spatial Pyramid Pooling (ASPP in Deeplab)
- Pyramid Pooling Modules (PPM in PSPNet)
- Gaussian Dynamic Convolution

### Model

We also develop the training and testing process for some published segmentation models:

- Deeplab v3+
- PSPNet

### Optimization

There are some training tricks in this repo to boost the performance:

- OHEM Loss
- Poly learning rate schedules

### Visualization

In order to visualize the training and testing process. We develop a Logger module. It can log training loss and can use the Tensorboard to visualize the scalar curve and the image.

---

## Performance
We evaluate the performance (mIoU) of the models in this repo on Cityscapes validation set.

| Model | Papers | Ours |
| ------ | ------ | ------ |
| PSPNet | 77.48 | 77.29 |
| DeeplabV3+ | - | 79.16 |

---

## TODO List

- **Datasets**
    - [ ] Pascal VOC 2012
    
- **Modules**
    - [ ] Object Context Module

- **Visualization and Other Utils**
    - [ ] PR-Curve
    - Segmentation Result Colorize
        - [ ] ADE20K
        - [ ] Pascal Context
        - [ ] COCO