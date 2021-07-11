from easydict import EasyDict as edict
from torchsegmentor.dataset.transforms.transforms import (
    HorizontalFlip,
    RandomCrop,
    ToTensor,
    Normalize,
)
from torchvision.transforms import Compose


cfg = edict()

cfg.train = edict()
cfg.val = edict()

cfg.seed = 307

# ======= Dataset cfg =======
cfg.num_workers = 8
cfg.dataset = "cityscapes"
cfg.root_path = "/root/data/datasets/cityscapes"
cfg.n_class = 19


# ======= Log cfg =======
cfg.log_dir = None
cfg.use_tensorboard = True
cfg.print_interval = 50

# ======= Network cfg =======
cfg.backbone = "resnet101"
cfg.bn = "inplace_abn_sync"
cfg.out_stride = 8

# ======= Distributed cfg =======
cfg.distributed = edict()
cfg.distributed.world_size = 1
cfg.distributed.gpu_id = -1
cfg.distributed.rank = 0
cfg.distributed.ngpus_per_node = 1
cfg.distributed.dist_url = "tcp://127.0.0.1:34568"

# ======= Train cfg =======
cfg.train.batch_size = 4
cfg.train.iter_num = 40000
cfg.train.lr = 1e-2
cfg.train.momentum = 0.9
cfg.train.weight_decay = 5e-4
cfg.train.power = 0.9
cfg.train.transforms = Compose(
    [
        HorizontalFlip(),
        RandomCrop((768, 768)),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# ======= Evaluation cfg =======
cfg.eval_mode = "val"
cfg.save_output = False
cfg.output_dir = "./output/"

cfg.val.batch_size = 2
cfg.val.flip = True
cfg.val.scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
cfg.val.transforms = Compose(
    [ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
)
