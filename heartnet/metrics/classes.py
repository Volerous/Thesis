import tensorflow as tf
from tensorflow.keras.metrics import *
from tensorflow.python.keras.metrics import MeanMetricWrapper
from mpunet.evaluate.metrics import one_class_dice, sparse_fg_recall, sparse_fg_precision, sparse_mean_fg_precision, sparse_mean_fg_recall

# From https://github.com/perslev/MultiPlanarUNet with modifications
class Dice(MeanMetricWrapper):
    def __init__(self, name="dice", dtype=None, **kwargs):
        def fn(true, pred):
            pred = tf.cast(tf.argmax(pred, axis=-1), tf.float32)
            return one_class_dice(tf.squeeze(true), pred)

        super().__init__(fn, name=name, dtype=dtype, **kwargs)


class FGRecall(Metric):
    def __init__(self, name="fg_recall", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.sum = self.add_weight("sum", initializer="zero")
        self.count = self.add_weight("count", initializer="zero")

    def update_state(self, true, pred, **kwargs):
        pred = tf.argmax(pred, axis=-1)

        # Get confusion matrix
        cm = tf.math.confusion_matrix(tf.reshape(true, [-1]),
                                      tf.reshape(pred, [-1]))
        if tf.size(cm) > 1:
            cm = tf.cast(cm, tf.float32)
            # Get precisions
            TP = tf.linalg.diag_part(cm)
            recalls = tf.math.divide_no_nan(TP, tf.math.reduce_sum(cm, axis=1))

            self.sum.assign_add(tf.math.reduce_mean(recalls[1:]))
            self.count.assign_add(1)

    def result(self):
        return self.sum / self.count

    def reset_states(self):
        self.sum.assign(0)
        self.count.assign(0)


class FGPrecision(Metric):
    def __init__(self, name="fg_precision", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.sum = self.add_weight("sum", initializer="zero")
        self.count = self.add_weight("count", initializer="zero")

    def update_state(self, true, pred, **kwargs):
        pred = tf.argmax(pred, axis=-1)

        # Get confusion matrix
        cm = tf.math.confusion_matrix(tf.reshape(true, [-1]),
                                      tf.reshape(pred, [-1]))
        if tf.size(cm) > 1:
            cm = tf.cast(cm, tf.float32)
            # Get precisions
            TP = tf.linalg.diag_part(cm)
            precs = tf.math.divide_no_nan(TP, tf.math.reduce_sum(cm, axis=0))

            self.sum.assign_add(tf.math.reduce_mean(precs[1:]))
            self.count.assign_add(1)

    def result(self):
        return self.sum / self.count

    def reset_states(self):
        self.sum.assign(0)
        self.count.assign(0)


class FGF1Score(Metric):
    def __init__(self, name="fg_f1", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.sum = self.add_weight("sum", initializer="zero")
        self.count = self.add_weight("count", initializer="zero")

    def update_state(self, true, pred, **kwargs):
        y_pred = tf.argmax(pred, axis=-1)

        # Get confusion matrix
        cm = tf.math.confusion_matrix(tf.reshape(true, [-1]),
                                      tf.reshape(y_pred, [-1]))
        if tf.size(cm) > 1:
            cm = tf.cast(cm, tf.float32)
            # Get precisions
            TP = tf.linalg.diag_part(cm)
            precisions = tf.math.divide_no_nan(TP,
                                               tf.math.reduce_sum(cm, axis=0))
            # Get recalls
            TP = tf.linalg.diag_part(cm)
            recalls = tf.math.divide_no_nan(TP, tf.math.reduce_sum(cm, axis=1))

            # Get F1s
            f1s = tf.math.divide_no_nan((2 * precisions * recalls),
                                        (precisions + recalls))
            self.sum.assign_add(tf.math.reduce_mean(f1s[1:]))
            self.count.assign_add(1)

    def result(self):
        return self.sum / self.count

    def reset_states(self):
        self.sum.assign(0)
        self.count.assign(0)