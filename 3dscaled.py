#!/usr/bin/env python3
from mpunet.models import UNet3D
from tensorflow import keras
from heartnet.layers.RobustScaler import RobustScaler
from heartnet.models.base import BaseModelTraining
from sklearn.preprocessing import robust_scale
for i in range(3):
    base3D = BaseModelTraining(keras.Sequential(
        [keras.Input(shape=(112, 112, 112, 1)),
            RobustScaler(),
        UNet3D(2, dim=112, out_activation="softmax")]),
                            name=f"pad-robust-scaled-{i}")
    base3D.setup()
    base3D.train()
    base3D.evaluate(True)
# rs = RobustScaler()
# for x, y in base3D._train_ds:
#     print(rs(x, y)[0].numpy().max())
