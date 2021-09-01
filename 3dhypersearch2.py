#!/usr/bin/env python3
from mpunet.models import UNet3D
from heartnet.models.base import BaseModelTraining
from heartnet.augmentation.elastic import Elastic3D
sigma_lower = 20
sigma = 30
cf = 1
depth = 5
for cf in [2, 3]:
    for depth in [3, 4]:
        for i in range(3):
            base3D = BaseModelTraining(
                UNet3D(
                    2,
                    dim=96,
                    out_activation="softmax",
                    complexity_factor=cf,
                    depth=depth
                ),
                name=f"hyper-{sigma_lower}-{sigma}-{cf}-{depth}-{i}"
            )
            base3D.batch_size = 1
            base3D.augmentations = [
                Elastic3D(
                    alpha=[0, 100],
                    sigma=[sigma_lower, sigma],
                    apply_prob=0.333
                )
            ]
            base3D.aug_repeats = 0
            base3D.setup()
            base3D.train()
            base3D.evaluate()