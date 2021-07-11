import os
from PIL import Image
from .base import BaseDataset


class ADE20K(BaseDataset):
    def __init__(self, root_path, mode='train', transforms=None, *args, **kwargs):
        super(ADE20K, self).__init__(
            root_path, mode, transforms, *args, **kwargs)

        self.mode = mode
        self.root = root_path
        assert os.path.join(self.root), "The path doesn't exist!"
        self.data_idx = self.load_data_idx()

    def __getitem__(self, index):
        if self.mode == 'test':
            imagename = self.data_idx[index]
            img = Image.open(imagename).convert('RGB')
            return self.transforms((img,)), None, imagename.split('/')[-1]

        imagename, labelname = self.data_idx[index]
        img = Image.open(imagename).convert('RGB')
        label = Image.open(labelname)

        if self.transforms is not None:
            img, label = self.transforms((img, label))
        label = label - 1
        label = label[0, :, :].long()
        return img, label, labelname.split('/')[-1]

    def load_data_idx(self):
        def get_path_pairs(img_folder, mask_folder):
            img_paths = []
            mask_paths = []
            for filename in os.listdir(img_folder):
                basename, _ = os.path.splitext(filename)
                if filename.endswith('.jpg'):
                    imgpath = os.path.join(img_folder, filename)
                    maskname = basename + '.png'
                    maskpath = os.path.join(mask_folder, maskname)
                    if os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask:', maskpath)
            return zip(img_paths, mask_paths)

        if self.mode == 'train':
            img_folder = os.path.join(self.root, 'images/training')
            mask_folder = os.path.join(self.root, 'annotations/training')
            data_idx = get_path_pairs(img_folder, mask_folder)
            assert len(data_idx) == 20210

        elif self.mode == 'val':
            img_folder = os.path.join(self.root, 'images/validation')
            mask_folder = os.path.join(self.root, 'annotations/validation')
            data_idx = get_path_pairs(img_folder, mask_folder)
            assert len(data_idx) == 2000

        elif self.mode == 'test':
            folder = os.path.join(self.root, '../release_test')
            with open(os.path.join(self.root, 'list.txt')) as f:
                data_idx = [os.path.join(folder, 'testing', line.strip()) for line in f]
            assert len(img_paths) == 3352

        return data_idx






