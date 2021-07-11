import random

import torchsegmentor.dataset.transforms.functional as F


class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() > self.p:
            return data
        else:
            transformed_data = []
            for item in data:
                transformed_data.append(F.flip(item))
            return tuple(transformed_data)


class ToTensor(object):
    def __call__(self, data):
        transformed_data = []
        for item in data:
            transformed_data.append(F.to_tensor(item))
        return tuple(transformed_data)


class Resize(object):
    def __init__(self, size, resize_mode=(0, 1)):
        self.w, self.h = size
        self.resize_mode = resize_mode

    def __call__(self, data):
        transformed_data = []
        for i, item in enumerate(data):
            transformed_data.append(
                F.resize(item, (self.w, self.h), self.resize_mode[i])
            )
        return tuple(transformed_data)


class Normalize(object):
    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1), norm_flags=(1, 0)):
        self.mean = mean
        self.std = std
        self.norm_flags = norm_flags

    def __call__(self, data):
        transformed_data = []
        for i, item in enumerate(data):
            if self.norm_flags[i] == 1:
                item = F.normalize(item, self.mean, self.std)
            transformed_data.append(item)
        return tuple(transformed_data)


class RandomCrop(object):
    def __init__(self, size, resize_mode=(0, 1)):
        self.w, self.h = size
        self.resize_mode = resize_mode

    def __call__(self, data):
        transformed_data = []
        w, h = F.get_size(data[0])
        if (self.w, self.h) == (w, h):
            return tuple(data)

        scale_flag = w < self.w or h < self.h
        if scale_flag:
            scale = float(self.w) / w if w < h else float(self.h) / h
            w, h = int(scale * w + 1), int(scale * h + 1)

        sw, sh = random.random() * (w - self.w), random.random() * (h - self.h)
        crop_size = int(sw), int(sh), int(sw) + self.w, int(sh) + self.h

        for i, item in enumerate(data):
            if scale_flag:
                item = F.resize(item, (w, h), self.resize_mode[i])
            transformed_data.append(F.crop(item, crop_size))

        return tuple(transformed_data)


class RandomScale(object):
    def __init__(self, scales=(1,), resize_mode=(0, 1)):
        self.scales = scales
        self.resize_mode = resize_mode

    def __call__(self, data):
        data_type = F.check_data_type(data[0])
        #w, h = F.get_size(data[0], data_type)
        w, h = F.get_size(data[0])
        scale = random.choice(self.scales)
        w, h = int(w * scale), int(h * scale)

        transform_data = []
        for i, item in enumerate(data):
            data_type = F.check_data_type(item)
            transform_data.append(
                F.resize(item, (w, h), self.resize_mode[i])
            )
        return tuple(transform_data)


class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5):
        self.brightness = [max(1 - brightness, 0), 1 + brightness]
        self.contrast = [max(1 - contrast, 0), 1 + contrast]
        self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, data):
        data_type = F.check_data_type(data[0])
        if data_type == 0:
            transform_data = []
            for i, item in enumerate(data):
                data_type = F.check_data_type(item)
                if data_type == 0:
                    item = F.color_jitter(
                        item, self.brightness, self.contrast, self.saturation
                    )
                transform_data.append(item)
            return tuple(transform_data)
        else:
            raise ValueError("ColorJitter only support PIL image.")


class RandomRotate(object):
    def __init__(self):
        self.angle = random.randint(1, 360)

    def __call__(self, data):
        data_type = F.check_data_type(data[0])
        transform_data = []
        for i, item in enumerate(data):
            data_type = F.check_data_type(item)
            transform_data.append(F.random_rotate(item, self.angle))
        return tuple(transform_data)
