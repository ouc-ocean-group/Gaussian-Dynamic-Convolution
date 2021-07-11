from torchsegmentor.operator import DistributedOperator

# ==== import your configuration here ===
from configs.deeplabv3p_resnet101_cityscapes import cfg

# ==== import your model operator here ===
from torchsegmentor.operator import DeeplabV3PlusOperator


if __name__ == "__main__":
    dis_operator = DistributedOperator(cfg, DeeplabV3PlusOperator)
    dis_operator.train()
    print("=> Training is Done!")
