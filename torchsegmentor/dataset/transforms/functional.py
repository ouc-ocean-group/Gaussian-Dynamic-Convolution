import torch
import PIL
from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import cv2
import numpy as np
import torchvision.transforms.functional as tvF
import torch.nn.functional as torchF
import random


def is_PIL(obj):
    return (
        isinstance(obj, PIL.Image.Image)
        or isinstance(obj, PIL.PngImagePlugin.PngImageFile)
        or isinstance(obj, PIL.JpegImagePlugin.JpegImageFile)
    )


def is_numpy(obj):
    return isinstance(obj, (np.ndarray, np.generic))


def is_tensor(obj):
    return isinstance(obj, torch.Tensor)


def check_data_type(obj):
    if is_PIL(obj):
        return 0
    elif is_numpy(obj):
        return 1
    elif is_tensor(obj):
        return 2
    else:
        raise TypeError


def flip(data):
    """Flip the input data.

    Arguments:
        data {PIL Image, numpy array, or tensor} -- data

    """
    data_type = check_data_type(data)
    if data_type == 0:
        return data.transpose(Image.FLIP_LEFT_RIGHT)
    elif data_type == 1:
        return cv2.flip(data, 1)
    elif data_type == 2:
        return torch.flip(data, dims=(2,))


def to_tensor(data):
    """Transform the input data to torch tensor.

    Arguments:
        data {PIL Image, numpy array, or tensor} -- data

    """
    data_type = check_data_type(data)
    if data_type == 0:
        return tvF.to_tensor(data)
    elif data_type == 1:
        data = torch.from_numpy(data)
        if data.dim() == 3:
            data = data.permute(2, 0, 1)
        else:
            data = data.unsqueeze(0)
        return data
    elif data_type == 2:
        return data


def get_size(data):
    """Return the size of the input data.

    Arguments:
        data {PIL Image, numpy array, or tensor} -- data

    """
    data_type = check_data_type(data)
    if data_type == 0:
        return data.size
    elif data_type == 1:
        return data.shape[1], data.shape[0]
    elif data_type == 2:
        return data.size(2), data.size(1)


RESIZE_MODE_OPTION = {
    0: {0: Image.BILINEAR, 1: Image.NEAREST},
    1: {0: cv2.INTER_LINEAR, 1: cv2.INTER_NEAREST},
    2: {0: "bilinear", 1: "nearest"},
}


def resize(data, size, mode=0):
    """Scale the input data.

    Arguments:
        data {PIL Image, numpy array, or tensor} -- data
        size {tuple} -- w, h

    Keyword Arguments:
        mode {int} -- {0:bilinear, 1:nearest}
    """
    data_type = check_data_type(data)
    mode = RESIZE_MODE_OPTION[data_type][mode]
    if data_type == 0:
        return data.resize((size[0], size[1]), mode)
    elif data_type == 1:
        return cv2.resize(data, (size[0], size[1]), interpolation=mode)
    elif data_type == 2:
        if mode == "nearest":
            data = torchF.interpolate(
                data.unsqueeze(0), size=(size[1], size[0]), mode=mode
            )
        else:
            data = torchF.interpolate(
                data.unsqueeze(0),
                size=(size[1], size[0]),
                mode=mode,
                align_corners=True,
            )
        return data.view(data.size(1), data.size(2), data.size(3))


def crop(data, size):
    """Scale the input data.

    Arguments:
        data {PIL Image, numpy array, or tensor} -- data
        size {tuple} -- w, h

    """
    data_type = check_data_type(data)
    if data_type == 0:
        return data.crop(size)
    elif data_type == 1:
        return data[size[1] : size[3], size[0] : size[2]]
    elif data_type == 2:
        return data[:, size[1] : size[3], size[0] : size[2]]


def normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalize the input data.

    Arguments:
        data {tensor} -- data
        size {tuple} -- w, h
        mean {tuple} -- mean value
        std {tuple} -- std

    """
    assert is_tensor(data)
    return tvF.normalize(data, mean, std)


def denormalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalize the input data.

    Arguments:
        data {tensor} -- data
        size {tuple} -- w, h
        mean {tuple} -- mean value
        std {tuple} -- std

    """
    assert is_tensor(data)
    data = tvF.normalize(data, mean=[0.0, 0.0, 0.0], std=[1 / x for x in std])
    data = tvF.normalize(data, mean=[-1 * x for x in mean], std=[1.0, 1.0, 1.0])
    return data


def color_jitter(data, brightness, contrast, saturation):
    """Color Jitter the input data

    Arguments:
        data {PIL Image} -- data

    Keyword Arguments:
        brightness {list} -- how much to jitter brightness
        contrast {list} -- how much to jitter contrast
        saturation {list} -- how much to jitter saturation
        data_type {data type} -- which module should be used to resize the data. {0:PIL, 1:cv2, 2:torch} (default: {0})

    """
    assert is_PIL(data)
    r_brightness = random.uniform(brightness[0], brightness[1])
    r_contrast = random.uniform(contrast[0], contrast[1])
    r_saturation = random.uniform(saturation[0], saturation[1])
    im = ImageEnhance.Brightness(data).enhance(r_brightness)
    im = ImageEnhance.Contrast(im).enhance(r_contrast)
    im = ImageEnhance.Color(im).enhance(r_saturation)

    return im


def random_rotate(data, angle):
    """Random Rotate the input data

    Arguments:
        data {PIL Image, numpy array, or tensor} -- data
        angle {int} -- the random image rotation angle

    """
    data_type = check_data_type(data)
    if data_type == 0:
        data = data.rotate(angle)
        return data

    elif data_type == 1:
        (h, w) = data.shape[:2]
        RotateMatrix = cv2.getRotationMatrix2D(
            center=(data.shape[1] / 2, data.shape[0] / 2), angle=angle, scale=1
        )
        RotImg = cv2.warpAffine(data, RotateMatrix, (w, h))
        return RotImg

    elif data_type == 2:
        a = data.numpy()
        (h, w) = a.shape[:2]
        RotateMatrix = cv2.getRotationMatrix2D(
            center=(a.shape[1] / 2, a.shape[0] / 2), angle=angle, scale=1
        )
        RotImg = cv2.warpAffine(a, RotateMatrix, (w, h))
        b = torch.from_numpy(RotImg)
        return b
