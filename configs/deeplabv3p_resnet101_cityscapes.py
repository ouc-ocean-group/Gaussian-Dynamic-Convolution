from easydict import EasyDict as edict
from torchsegmentor.dataset.transforms.transforms import (
    HorizontalFlip,
    ColorJitter,
    RandomScale,
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
cfg.root_path = "./data/cityscapes"
cfg.n_class = 19

# ======= Log cfg =======
cfg.save_log = False
cfg.log_dir = "./log/dv3"
cfg.use_tensorboard = True
cfg.print_interval = 50
cfg.print_img_num=4100

# ======= Network cfg =======
cfg.backbone = "resnet101"
#cfg.backbone = "resnet50"
cfg.sync_bn = True
cfg.aspp_type = "normal"
cfg.out_stride = 16

# ======= Distributed cfg =======
cfg.distributed = edict()
cfg.distributed.world_size = 1
cfg.distributed.gpu_id = -1
cfg.distributed.rank = 0
cfg.distributed.ngpus_per_node = 1
cfg.distributed.dist_url = "tcp://127.0.0.1:34568"
cfg.distributed.mode="train"

# ======= Train cfg =======
cfg.train.batch_size = 16
cfg.train.iter_num = 41000
cfg.train.lr = 1e-2
cfg.train.momentum = 0.9
cfg.train.weight_decay = 5e-4
cfg.train.power = 0.9
cfg.train.eval_during_training=True
cfg.train.eval_step= 41000
cfg.train.transforms = Compose(
    [
        HorizontalFlip(),
        ColorJitter(),
        RandomScale((0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
        RandomCrop((768, 768)),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# ======= Evaluation cfg =======
cfg.eval_mode = "val"
cfg.eval_model_path = "./log/dv3/model.pth"
cfg.save_output = True
cfg.output_dir = "./output/"

cfg.val.batch_size = 4
cfg.val.flip = True
cfg.val.scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
cfg.val.transforms = Compose(
    [ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
)
