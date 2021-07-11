from PIL import Image
import numpy as np
import scipy.io
import os.path as osp
from .base import BaseDataset


class VOCContext(BaseDataset):
    def __init__(self, root_path, subset='train', transforms=None, *args, **kwargs):
        super(VOCContext, self).__init__(
            root_path, subset, transforms, *args, **kwargs)

        valid_classes = [
            0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22,
            23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296,
            427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424,
            68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360,
            98, 187, 104, 105, 366, 189, 368, 113, 115]
        self.label_map = dict(zip(valid_classes, range(len(valid_classes))))

    def load_data_idx(self):
        imgnames = []
        img_set_list_path = osp.join(
            self.root_path, 'ImageSets/Main/{}.txt'.format(self.subset))
        with open(img_set_list_path, 'r') as reader:
            data_list = reader.readlines()
        data_idx = [[name.strip()+'.jpg', name.strip()+'.mat']
                    for name in data_list]
        return data_idx

    def __getitem__(self, idx):
        img_name, label_name = self.data_idx[idx]
        img = Image.open(
            osp.join(self.root_path, 'JPEGImages', img_name)).convert("RGB")
        label = scipy.io.loadmat(osp.join(self.root_path, 'context', label_name))[
            'LabelMap'].astype(np.int32)
        if self.transforms is not None:
            img, label = self.transforms((img, label))
        label = self.convert_labels(label)
        label = label[0, :, :].long()
        return img, label, label_name.split('/')[-1]
