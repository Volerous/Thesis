#!/usr/bin/env python3
from mpunet.models import UNet
from heartnet.models.base import BaseModelTraining
from tensorflow.keras import *
for i in range(3):
    base = BaseModelTraining(
        UNet(2, depth=4, dim=128, out_activation="softmax"), name=f"match3Dpad{i}"
    )
    base.batch_size = 111
    base.setup()
    base.train()
    base.evaluate()