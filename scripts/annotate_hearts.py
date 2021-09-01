#!/usr/bin/env python3
from matplotlib import colors
from heartnet.augmentation import Elastic2D
from mpunet.models import UNet
from heartnet.models.base import BaseModelTraining
from tensorflow.keras import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
os.chdir("/homes/pmcd/Peter_Patrick3/final_images")


def squeeze(x, y):
    return tf.squeeze(x), tf.squeeze(y)


# augs = itertools.product([0, 450], [0, 30])
# for k, (a, s) in enumerate(augs):
#     print(k+1,a,s)
#     base = BaseModelTraining(
#         UNet(
#             2, depth=4, dim=128, out_activation="softmax", complexity_factor=2
#         ), f"base"
#     )
#     base.augmentations = [Elastic2D(alpha=a, sigma=s, apply_prob=1.0),]
#     base.setup()

#     ds = base._final_ds.unbatch().batch(111)
#     for idx, (x, y) in enumerate(ds.map(squeeze), start=44):
#         os.makedirs(f"anon{idx}/aug{k+1}",exist_ok=True)
#         for j in range(111):
#             plt.imshow(x[j], cmap="gray")
#             plt.imshow(
#                 np.ma.masked_where(y[j] == 0, y[j]),
#                 cmap=plt.cm.get_cmap("gray").set_under(alpha=0.0),
#                 alpha=0.5
#             )
#             plt.savefig(f"anon{idx}/aug{k+1}/{j}.jpg")
#             plt.clf()
base = BaseModelTraining(
    UNet(2, depth=1, dim=128, out_activation="softmax", complexity_factor=2),
    f"base"
)
base.setup()

ds = base._final_ds.unbatch().batch(111)
for idx, (x, y) in enumerate(ds.map(squeeze), start=44):
    os.makedirs(f"anon{idx}/base",exist_ok=True)
    for j in range(111):
        print(tf.math.maximum(x[j]))
        # plt.imshow(x[j], cmap="gray")
        # plt.imshow(
        #     np.ma.masked_where(y[j] == 0, y[j]),
        #     cmap=plt.cm.get_cmap("gray").set_under(alpha=0.0),
        #     alpha=0.5
        # )
        # plt.savefig(f"anon{idx}/base/{j}.jpg")
        # plt.clf()
