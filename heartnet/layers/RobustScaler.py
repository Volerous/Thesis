from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp
import tensorflow as tf


class RobustScaler(Layer):
    def __init__(self, unit_variance=False):
        self.q_min = 25.0
        self.q_max = 75.0
        self.qs = tf.convert_to_tensor([25, 50, 75])
        self.unit_variance = unit_variance
        super(RobustScaler, self).__init__()

    def call(self, X):
        qs = tfp.stats.percentile(X, self.qs)
        scale = qs[2] - qs[0]
        if self.unit_variance:
            adjust = (tfp.distributions.Normal(0,1).quantile(qs[2] / 100.0) -
                      tfp.distributions.Normal(0,1).quantile(qs[0] / 100.0))
            scale = scale / adjust
        return (X - qs[1])/ scale