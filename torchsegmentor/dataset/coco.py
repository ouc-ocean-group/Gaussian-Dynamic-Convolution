import os.path as osp
from PIL import Image
import numpy as np
from torchsegmentor.dataset.base import BaseDataset


# from pycocotools.coco import COCO as COCOApi
# from pycocotools import mask


class COCO(BaseDataset):

    def __init__(self, root_path, subset='train', transforms=None, *args, **kwargs):
        if subset in ['train', 'val']:
            self.coco = COCOApi(osp.join(root_path, 'annotations', 'instances_{}2017.json'.format(subset)))
        self.coco_mask = mask
        self.valid_cls = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                          1, 64, 20, 63, 7, 72]
        super(COCO, self).__init__(root_path, subset, transforms, *args, **kwargs)

    def load_data_idx(self):
        ann_ids = list(self.coco.anns.keys())
        img_id = [self.coco.anns[ann_id]['image_id'] for ann_id in ann_ids]
        del ann_ids
        return img_id

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.valid_cls:
                c = self.valid_cls.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def __getitem__(self, idx):
        img_id = self.data_idx[idx]

        img_info = self.coco.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        img = Image.open(osp.join(self.root_path, '{}2017'.format(self.subset), img_name)).convert("RGB")

        cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        label = Image.fromarray(self._gen_seg_mask(cocotarget, img_info['height'], img_info['width']))

        if self.transforms is not None:
            img, label = self.transforms((img, label))

        return img, label
