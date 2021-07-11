import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler

from .voc_context import VOCContext
from .cityscapes import CityScapes
from .coco import COCO

from torchsegmentor.dataset.dataloader import Dataloader


datasets = {"voc_context": VOCContext, "cityscapes": CityScapes, "coco": COCO}


def make_dataloader(cfg, mode="train"):
    if cfg.dataset not in datasets:
        raise NotImplementedError(
            "{} dataset is not implemented for now.".format(cfg.dataset)
        )

    transforms = cfg[mode].transforms

    dataset = datasets[cfg.dataset](cfg.root_path, mode, transforms=transforms)

    cfg.num_workers = int(cfg.num_workers / cfg.distributed.world_size)
    cfg[mode].batch_size = int(cfg[mode].batch_size / cfg.distributed.world_size)

    data_sampler = DistributedSampler(dataset)
    data_loader = Dataloader(
        data.DataLoader(
            dataset,
            batch_size=cfg[mode].batch_size,
            num_workers=cfg.num_workers,
            sampler=data_sampler,
        )
    )

    if cfg.distributed.gpu_id == 0:
        print("=> {} {} subset with {} images.".format(cfg.dataset, mode, len(dataset)))
    return data_loader
