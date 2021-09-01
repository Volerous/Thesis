#!/usr/bin/env python3
from mpunet.models import UNet3D
from heartnet.models.base import BaseModelTraining
for i in range(3):
    base3D = BaseModelTraining(
        UNet3D(2, dim=112, out_activation="softmax"), name=f"pad-data-{i}", full=True
    )
    base3D.batch_size = 1
    base3D.setup()
    base3D.train()
    base3D.evaluate()
    base3D = BaseModelTraining(
        UNet3D(2, dim=96, out_activation="softmax"), name=f"shrink-data-{i}", full=True
    )
    base3D.batch_size = 1
    base3D.setup()
    base3D.train()
    base3D.evaluate()