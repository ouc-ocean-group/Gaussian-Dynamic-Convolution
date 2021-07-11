import torch
import torch.nn as nn


class OHEMCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        threshold,
        min_n,
        weight=None,
        size_average=None,
        ignore_index=255,
        reduce=None,
        reduction="mean",
        *args,
        **kwargs
    ):
        super(OHEMCrossEntropyLoss, self).__init__()
        self.default_ce_loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
        )

        self.threshold = threshold
        self.min_n = min_n
        self.ignore_index = ignore_index

    def forward(self, logits, label):
        with torch.no_grad():
            prob = torch.softmax(logits, dim=1)
            prob = prob.permute(1, 0, 2, 3).contiguous().view(prob.size(1), -1)
            label = label.view(-1)
            invalid_idx = label == self.ignore_index
            tmp_label = label
            tmp_label[invalid_idx] = 0
            label_prob = prob[tmp_label, torch.arange(tmp_label.size(0))]
            label_prob[invalid_idx] = 1
            if (label_prob <= self.threshold).float().sum() >= self.min_n:
                threshold = self.threshold
            else:
                bottom_prob, _ = torch.topk(label_prob, self.min_n, largest=False)
                threshold = bottom_prob[-1].item()
            invalid_idx = label_prob > threshold
            label[invalid_idx] = self.ignore_index
            label = label.reshape(
                logits.size(0), logits.size(2), logits.size(3)
            ).clone()
        loss = self.default_ce_loss(logits, label)
        return loss


if __name__ == "__main__":
    logits = torch.randn(2, 4, 100, 100)
    label = torch.randint(4, (2, 100, 100))
    label[:, 5, 5] = 255

    loss_func = OHEMCrossEntropyLoss(threshold=0.1, min_n=5000, ignore_index=255)

    loss = loss_func(logits, label)
    print(loss)
