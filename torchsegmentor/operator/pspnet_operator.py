import random
import torch
from torchsegmentor.operator.base_operator import BaseOperator

from torchsegmentor.model.pspnet import make_pspnet


class PSPNetOperator(BaseOperator):
    def __init__(self, cfg):
        self.cfg = cfg

        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        pspnet = make_pspnet(cfg).cuda(cfg.distributed.gpu_id)

        super(PSPNetOperator, self).__init__(cfg, pspnet)
