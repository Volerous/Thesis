#!/usr/bin/env python3
from mpunet.models import UNet3D
from heartnet.models.base import BaseModelTraining
from heartnet.augmentation.elastic import Elastic3D
for alpha in [25, 100, 200]:
    for prob in [0.5, 0.333]:
        for dim in [96, 112]:
            for i in range(3):
                base3D = BaseModelTraining(
                    UNet3D(2, dim=dim, out_activation="softmax"),
                    name=f"aug-{dim}-{prob}-{alpha}-{i}"
                )
                base3D.batch_size = 1
                base3D.augmentations = [
                    Elastic3D(
                        alpha=[0, alpha], sigma=[20, 30], apply_prob=prob
                    )
                ]
                base3D.aug_repeats = 0
                base3D.setup()
                base3D.train()
                base3D.evaluate()