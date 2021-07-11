import os
import torch.nn.functional as F
import random
import torch.optim as optim
import torch
import cv2
import numpy as np

from torchsegmentor.dataset import make_dataloader
from torchsegmentor.utils.optim.lr_scheduler import PolyLR
from torchsegmentor.utils.optim.loss import OHEMCrossEntropyLoss
from torchsegmentor.utils.record.recorder import Recorder, get_lr
from torchsegmentor.utils.metric import Metrics
import torchsegmentor.dataset.transforms.functional as transF
import torchsegmentor.utils.vis.color_map as cm
from torchsegmentor.utils.vis.visualize import colorize


class BaseOperator(object):
    def __init__(self, cfg, model, optimizer=None, lr_sch=None,flag=True):
        self.cfg = cfg

        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if not flag:
            model.module.load_state_dict(torch.load(self.cfg.eval_model_path))
        self.model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.cfg.distributed.gpu_id]
        )



        self.optimizer = (
            optim.SGD(
                self.model.parameters(),
                lr=cfg.train.lr,
                momentum=cfg.train.momentum,
                weight_decay=cfg.train.weight_decay,
            )
            if optimizer is None
            else optimizer
        )

        self.poly_lr = (
            PolyLR(self.optimizer, self.cfg.train.iter_num, self.cfg.train.power)
            if lr_sch is None
            else lr_sch
        )

        self.loss = OHEMCrossEntropyLoss(
            threshold=0.7, min_n=int(100000 / cfg.distributed.world_size)
        )

        self.data_loader = make_dataloader(self.cfg)
        self.eval_data_loader = make_dataloader(self.cfg, self.cfg.eval_mode)

    def criterion(self, outs, labels):
        return self.loss(outs, labels)

    def training_process(self):
        recorder = (
            Recorder(
                print_interval=self.cfg.print_interval,
                use_tensorboard=self.cfg.use_tensorboard,
                save_log=self.cfg.save_log,
                log_dir=self.cfg.log_dir,
            )
            if self.cfg.distributed.gpu_id == 0
            else None
        )
        if recorder is not None:
            recorder.print_cfg(self.cfg)

        self.model.train()

        for iter_step in range(self.cfg.train.iter_num):
            self.optimizer.zero_grad()
            imgs, labels, _ = self.data_loader.get_batch()

            outs = self.model(imgs)

            loss = self.criterion(outs, labels)
            loss.backward()
            self.optimizer.step()

            self.poly_lr.step()

            if recorder is not None:
                scalar_data = {"Train/Loss": loss.item()}
                for lr_i, lr in enumerate(get_lr(self.optimizer)):
                    scalar_data["Train/LR{}".format(lr_i)] = lr
                recorder.record_scalar(iter_step,scalar_data)
                recorder.save_ckp(iter_step,self.model.module)
                if(
                        iter_step % self.cfg.print_img_num == self.cfg.print_img_num - 1
                ):
                    img, pred, gt = self.visualize(imgs, outs, labels, cm.CITYSCAPES)
                    recorder.record_img(iter_step, "Train", [img, pred, gt])

            if (
                self.cfg.train.eval_during_training
                and iter_step % self.cfg.train.eval_step == self.cfg.train.eval_step - 1
            ):
                self.evaluate_process(during_training=True)
                self.model.train()

    def evaluate_process(self, during_training=True):
        if not during_training:
            self.model.module.load_state_dict(torch.load(self.cfg.eval_model_path))
            #self.load_gpus(self.cfg.eval_model_path)
        self.model.eval()
        metrics = Metrics(self.cfg.n_class)
        data_num = len(self.eval_data_loader)
        print("data_num:{}".format(data_num))
        #self.eval_data_loader.sampler.set_epoch(0)
        self.eval_data_loader.fresh()
        with torch.no_grad():
            #for i, batch in enumerate(self.eval_data_loader):
            for i in range(data_num//self.cfg.val.batch_size):
                if i % data_num == data_num-1:
                    print("\r[{}/{}]".format(i, data_num), end="", flush=True)
                #imgs, labels, names = batch
                imgs,labels,names=self.eval_data_loader.get_batch()

                n, _, h, w = imgs.size()

                imgs = imgs.cuda(self.cfg.distributed.gpu_id)
                probs = torch.zeros((n, self.cfg.n_class, h, w)).cuda(
                    self.cfg.distributed.gpu_id
                )

                for scale in self.cfg[self.cfg.eval_mode].scales:
                    scaled_imgs = F.interpolate(
                        imgs,
                        size=(int(h * scale), int(w * scale)),
                        mode="bilinear",
                        align_corners=True,
                    )
                    outs = self.model(scaled_imgs)
                    #prob = torch.softmax(outs[0], dim=1).unsqueeze(1)
                    prob = torch.softmax(outs, dim=1)
                    #!!
                    '''prob = F.interpolate(
                        prob, (h, w), mode="bilinear", align_corners=True
                    ).view(n, h, w)'''
                    prob = F.interpolate(
                        prob, (h, w), mode="bilinear", align_corners=True
                    )
                    probs += prob

                    if self.cfg[self.cfg.eval_mode].flip:
                        outs = self.model(torch.flip(scaled_imgs, dims=(3,)))
                        outs = torch.flip(outs, dims=(3,))
                        prob = torch.softmax(outs, dim=1)
                        prob = F.interpolate(
                            prob, (h, w), mode="bilinear", align_corners=True
                        )
                        probs += prob
                probs = probs.cpu().numpy()
                preds = np.argmax(probs, axis=1)

                metrics.add_batch(preds, labels.cpu().numpy())

                if self.cfg.save_output:
                    for idx in range(n):
                        cv2.imwrite(
                            os.path.join(self.cfg.output_dir, names[idx]),
                            preds[idx, :, :],
                        )

            metrics.all_reduce()

        if self.cfg.distributed.gpu_id == 0:
            print("Performance:")
            print(
                "PixelAcc.: {:.6} | PixelClsAcc.: {:.6} | mIoU: {:.6} | pwmIoU: {:.6}".format(
                    metrics.Pixel_Accuracy(),
                    metrics.Pixel_Accuracy_Class(),
                    metrics.Mean_Intersection_over_Union(),
                    metrics.Frequency_Weighted_Intersection_over_Union(),
                )
            )

    @staticmethod
    def visualize(imgs, preds, labels, color_map):
        img = transF.denormalize(imgs[0].cpu()).unsqueeze(0)

        #pred = preds[0][0].detach().cpu().numpy()
        pred = preds[0].detach().cpu().numpy()
        pred = np.argmax(pred, axis=0)
        pred = torch.as_tensor(colorize(pred, color_map, normalize=True)).unsqueeze(0)

        label = labels[0].cpu().numpy()
        label = torch.as_tensor(colorize(label, color_map, normalize=True)).unsqueeze(0)
        return img, pred, label

    def load_gpus(self,model_path):
        static_dict = torch.load(model_path)
        #from collections import OrderedDict
        #new_static_dict = OrderedDict()
        #for k,v in static_dict.items():
        #    name = k[7:]
        #    new_static_dict[name]=v
        info=self.model.module.load_state_dict(static_dict)
