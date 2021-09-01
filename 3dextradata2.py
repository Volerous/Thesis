#!/usr/bin/env python3
from mpunet.models import UNet3D
from heartnet.models.base import BaseModelTraining
from heartnet.augmentation.elastic import Elastic3D
for i in range(3):
    # base3D = BaseModelTraining(
    #     UNet3D(2, dim=96, out_activation="softmax"), name=f"augmentation-data-{i}", full=True
    # )
    # base3D.batch_size = 1
    # base3D.augmentations = [
    #     Elastic3D(alpha=[0, 450], sigma=[20, 30], apply_prob=0.333)
    # ]
    # base3D.aug_repeats = 0
    # base3D.setup()
    # base3D.train()
    # base3D.evaluate()
    base3D = BaseModelTraining(
        UNet3D(2, dim=96, out_activation="softmax"), name=f"hyper-1-data-{i}", full=True
    )
    base3D.batch_size = 1
    base3D.augmentations = [
        Elastic3D(alpha=[0, 100], sigma=[20, 30], apply_prob=0.333)
    ]
    base3D.aug_repeats = 0
    base3D.setup()
    base3D.train()
    base3D.evaluate()
    base3D = BaseModelTraining(
        UNet3D(2, dim=96, depth=4, out_activation="softmax"), name=f"hyper-2-data-{i}", full=True
    )
    base3D.batch_size = 1
    base3D.augmentations = [
        Elastic3D(alpha=[0, 100], sigma=[20, 30], apply_prob=0.333)
    ]
    base3D.aug_repeats = 0
    base3D.setup()
    base3D.train()
    base3D.evaluate()