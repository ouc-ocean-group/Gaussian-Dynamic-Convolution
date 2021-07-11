import numpy as np
import torch
import torch.distributed as dist


class Metrics(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, pred, label):
        mask = (label >= 0) & (label < self.num_class)
        label = self.num_class * label[mask].astype("int") + pred[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, pred, label):
        assert label.shape == label.shape
        self.confusion_matrix += self._generate_matrix(pred, label)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def all_reduce(self):
        confusion_matrix = torch.from_numpy(self.confusion_matrix).cuda()
        dist.all_reduce(confusion_matrix, dist.ReduceOp.SUM)
        self.confusion_matrix = confusion_matrix.cpu().numpy().astype(np.float32)
