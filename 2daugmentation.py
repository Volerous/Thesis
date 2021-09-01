#!/usr/bin/env python3
from mpunet.models import UNet
from heartnet.models.base import BaseModelTraining
from heartnet.augmentation.elastic import Elastic2D

base = BaseModelTraining(
    UNet(2, depth=4, dim=128, out_activation="softmax"), "augmentation"
)
base.batch_size = 16
base.augmentations = [
    Elastic2D(alpha=[0, 450], sigma=[20, 30], apply_prob=0.333)
]
base.aug_repeats = 0
base.setup()
base.train()
base.evaluate()