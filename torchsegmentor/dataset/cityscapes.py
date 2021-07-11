import os.path as osp
from PIL import Image
import glob
import cv2
from .base import BaseDataset


class CityScapes(BaseDataset):
    def __init__(self, root_path, subset="train", transforms=None, *args, **kwargs):
        super(CityScapes, self).__init__(root_path, subset, transforms, *args, **kwargs)

    def load_data_idx(self):
        img_dir = osp.join(self.root_path, "leftImg8bit", self.subset)
        img_path = glob.glob(osp.join(img_dir, "*/*.png"))
        data_idx = [
            [
                osp.join(*path.split("/")[-2:]),
                osp.join(*path.split("/")[-2:]).replace(
                    "leftImg8bit", "gtFine_labelTrainIds"
                ),
            ]
            for path in img_path
        ]
        return data_idx

    def __getitem__(self, idx):
        img_name, label_name = self.data_idx[idx]

        img = Image.open(
            osp.join(self.root_path, "leftImg8bit", self.subset, img_name)
        ).convert("RGB")

        label = cv2.imread(osp.join(self.root_path, "gtFine", self.subset, label_name))

        if self.transforms is not None:
            img, label = self.transforms((img, label))
        label = label[0, :, :].long()
        return img, label, label_name.split("/")[-1]
