import torch.utils.data as data
import torch


class BaseDataset(data.Dataset):
    def __init__(self, root_path, subset="train", transforms=None):
        super(BaseDataset, self).__init__()

        self.root_path = root_path
        assert subset in ("train", "val", "test")
        self.subset = subset

        self.data_idx = self.load_data_idx()
        self.len = len(self.data_idx)

        self.transforms = transforms

    def load_data_idx(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return self.len

    def convert_labels(self, label):
        new_label = torch.ones_like(label) * 255
        for k, v in self.label_map.items():
            new_label[label == k] = v
        return new_label
