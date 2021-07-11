import torch
import random

from torchsegmentor.operator.base_operator import BaseOperator
from torchsegmentor.model.deeplabv3plus import make_deeplabv3plus


class DeeplabV3PlusOperator(BaseOperator):
    def __init__(self, cfg):
        self.cfg = cfg

        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        deeplabv3plus = make_deeplabv3plus(cfg).cuda(cfg.distributed.gpu_id)
        if cfg.distributed.mode=="eval":
            super(DeeplabV3PlusOperator, self).__init__(cfg, deeplabv3plus,flag=False)
        else:
            super(DeeplabV3PlusOperator, self).__init__(cfg, deeplabv3plus)
