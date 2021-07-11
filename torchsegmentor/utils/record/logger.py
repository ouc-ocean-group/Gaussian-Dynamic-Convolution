import tensorboardX as tbx
import os
import zipfile
from datetime import datetime
import shutil
import torch
import torchvision.utils as vutils
from torchsegmentor.utils.vis.timer import Timer


class Logger(object):
    def __init__(self, cfg):
        self.log_dir = self.create_log_dir(cfg.log_dir)
        self.save_cfg(cfg)
        self.tensorboard = (
            tbx.SummaryWriter(self.log_dir) if cfg.use_tensorboard else None
        )
        self.timer = Timer()
        self.init_timer(cfg.train.iter_num)

    # Basic function

    def init_timer(self, iter_length):
        self.timer.start(iter_length)

    def add_scalar(self, data, tag, n_iter):
        self.tensorboard.add_scalar(tag, float(data), n_iter)

    def add_img(self, data, tag, n_iter):
        self.tensorboard.add_image(tag, data, n_iter)

    def write_log_file(self, text):
        with open(os.path.join(self.log_dir, "log.txt"), "a+") as writer:
            writer.write(text)

    def parse_cfg(self, cfg_dict, father=""):
        all = ""
        if father != "":
            father += "."
        for k, v in cfg_dict.items():
            if isinstance(v, dict):
                all += self.parse_cfg(v, k)
            else:
                all += "{}{} = {}\n".format(father, k, v)
        return all

    # API
    def log(self, data, n_iter):
        """Log for training process
        
        Arguments:
            data {dict:{'scalar': {str: tensor}, 'imgs': {str: [tensor,]}} -- [loss, acc, miou etc. images for visualization]
            n_iter {int} -- [iteration step]
        """
        data_str = ""
        for k, v in data["scalar"].items():
            data_str += "{}: {:.6} ".format(k, float(v))
            self.add_scalar(float(v), tag=k, n_iter=n_iter)
        text = "{} Iter. {} | {}\n".format(self.timer.stamp(n_iter), n_iter, data_str)

        self.write_log_file(text)
        print(text)
        if "imgs" in data:
            for k, v in data["imgs"].items():
                vis_img = torch.cat(v, dim=0)
                vis_img = vutils.make_grid(vis_img, normalize=True, scale_each=True)
                self.add_img(vis_img, tag=k, n_iter=n_iter)

    def save_cfg(self, cfg):
        with open(os.path.join(self.log_dir, "cfg.txt"), "w") as writer:
            cfg_str = self.parse_cfg(cfg)
            print("\n===========================================")
            print("Configuration:")
            print(cfg_str)
            print("===========================================")
            writer.write(cfg_str)

    def save_model(self, model):
        print("Saving Model...")
        torch.save(model.state_dict(), os.path.join(self.log_dir, "model.pth"))

    @staticmethod
    def zip_dir(dir):
        tmp_path = dir.split("/")
        out_file_name = os.path.join(
            tmp_path[:-1],
            tmp_path[-1]
            + "-{}.zip".format(datetime.now().isoformat(timespec="seconds")),
        )
        zip = zipfile.ZipFile(out_file_name, "w", zipfile.ZIP_DEFLATED)
        for path, dirnames, filenames in os.walk(dir):
            fpath = path.replace(dir, "")
            for filename in filenames:
                zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
        zip.close()

    def create_log_dir(self, log_dir):
        if log_dir is not None:
            if os.path.isdir(log_dir):
                shutil.rmtree(log_dir)
        else:
            log_dir = os.path.join(
                "./log/{}".format(datetime.now().isoformat(timespec="seconds"))
            )
        os.makedirs(log_dir)
        return log_dir


class LossLogger(object):
    def __init__(self):
        self.keys = []
        self.loss_value = torch.zeros(1)
        self.step = 0

    def register_loss(self, loss_name):
        self.keys = loss_name
        self.loss_value = self.loss_value.expand(len(self.keys))

    def add_batch(self, loss_value):
        assert len(self.keys) == len(loss_value)
        self.step += 1
        for i, v in enumerate(loss_value):
            self.loss_value[i] += v

    def get_losses(self):
        loss = self.loss_value / self.step
        loss_kv = {self.keys[i]: v for i, v in enumerate(loss)}
        self.step = 0
        return loss_kv
