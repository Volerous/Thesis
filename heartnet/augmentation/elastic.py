from mpunet.augmentation import Elastic2D, Elastic3D
import numpy as np


class Elastic2D(Elastic2D):

    def __call__(self, x, y):
        return super().__call__(x, y, [0.0] * x.shape[-1])


class Elastic3D(Elastic3D):

    def __init__(self, alpha, sigma, apply_prob):
        super().__init__(alpha, sigma, apply_prob)

    def __call__(self, x, y, bg_val=0.0):
        augment_mask = np.random.rand(1) <= self.apply_prob
        if augment_mask:
            return self.trans_func(x, y, self.alpha, self.sigma, bg_val)
        return x, y