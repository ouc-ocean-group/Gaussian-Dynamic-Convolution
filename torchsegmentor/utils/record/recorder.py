import os
import shutil
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from .timer import Timer
from easydict import EasyDict
import torchvision.utils as tvutils
from torchvision.transforms import Compose


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group["lr"])
    if len(lrs) == 1:
        return lrs  # [0]
    else:
        return lrs


class Recorder(object):
    def __init__(
        self,
        print_interval=50,
        ckp_interval=5000,
        use_tensorboard=False,
        save_log=True,
        log_dir="./log",
    ):
        self.print_interval = print_interval
        self.ckp_interval = ckp_interval
        self.use_tensorboard = use_tensorboard
        self.save_log = save_log
        self.log_dir = log_dir

        if save_log:
            self.create_log_dir()

        self.tensorboard = SummaryWriter(self.log_dir) if use_tensorboard else None

        self.timer = Timer()
        self.timer.start(41000)

        self.scalar_data = {}
        self.data_count = 0

    def create_log_dir(self):
        try:
            os.makedirs(self.log_dir)
        except Exception as e:
            print("=> Cannot make log dir {} directly: {}".format(self.log_dir, str(e)))
            print("=> Trying to archive the old log dir...")
            if not os.path.isdir(self.log_dir):
                os.mkdir("./archive")
            shutil.move(self.log_dir, "./archive/{}".format(time.time()))

    def add_scalar_data(self, data):
        self.data_count += 1
        for k, v in data.items():
            assert isinstance(v, float)
            if k not in self.scalar_data:
                self.scalar_data[k] = v
            else:
                self.scalar_data[k] += v

    def avg_scalar_data(self):
        for k, v in self.scalar_data.items():
            self.scalar_data[k] = v / self.data_count

    def clear_scalar_data(self):
        for k, v in self.scalar_data.items():
            self.scalar_data[k] = 0

    def write_tb_scalar(self, data, tag, step):
        self.tensorboard.add_scalar(tag, float(data), step)

    def write_tb_img(self, data, tag, step):
        self.tensorboard.add_image(tag, data, step)

    def record_scalar(self, step, data):
        self.add_scalar_data(data)

        if step % self.print_interval == self.print_interval - 1:
            self.avg_scalar_data()
            line_holder = "{} ".format(self.timer.stamp(step))
            for k, v in self.scalar_data.items():
                line_holder += "{}: {:.4} ".format(k, v)
                if self.use_tensorboard:
                    self.write_tb_scalar(v, k, step)
            print(line_holder)

            if self.save_log:
                with open(os.path.join(self.log_dir, "log.txt"), "a+") as writer:
                    writer.write(line_holder + "\n")

            self.data_count = 0

    def record_img(self, step, tag, data):
        if self.use_tensorboard:
            img = torch.cat(data, dim=0)
            img = tvutils.make_grid(img, normalize=True, scale_each=True)
            self.write_tb_img(img, tag, step)

    def print_cfg(self, cfg, father_pre=""):
        if father_pre == "":
            print("==================== CONFIGURATION ==================")
        for i, (k, v) in enumerate(cfg.items()):
            if isinstance(v, EasyDict):
                print(father_pre + "[" + k + "]")
                self.print_cfg(v, father_pre=father_pre + "    ")
            elif isinstance(v, Compose):
                print(father_pre + "Transforms:")
                for t in v.transforms:
                    print(father_pre + "    " + t.__class__.__name__)
            else:
                print(father_pre + k + " " + "." * (25 - len(k)) + " " + str(v))
        if father_pre == "":
            print("=====================================================")

    def save_ckp(self, step, model):
        if step % self.ckp_interval == self.ckp_interval - 1:
            torch.save(
                model.state_dict(),
                os.path.join(self.log_dir, "ckp_{}.pth".format(step)),
            )
