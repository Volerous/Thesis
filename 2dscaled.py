#!/usr/bin/env python3
from mpunet.models import UNet
from heartnet.models.base import BaseModelTraining
from heartnet.layers.RobustScaler import RobustScaler
from tensorflow import keras
from tensorflow.keras import *
for i in range(3):
    base = BaseModelTraining(keras.Sequential([
        keras.Input(shape=(128, 128, 1)),
        RobustScaler(),
        UNet(2,
             depth=4,
             dim=128,
             out_activation="softmax",
             complexity_factor=2)
    ]),
                             f"base-data-robust-{i}",
                             full=True)
    base.batch_size = 32
    base.setup()
    base.train()
    base.evaluate()