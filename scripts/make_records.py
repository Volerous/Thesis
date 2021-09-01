import tensorflow as tf
from mpunet.models import UNet
from heartnet.models.base import BaseModelTraining
base = BaseModelTraining(
    UNet(2, depth=4, dim=128, out_activation="softmax", complexity_factor=2),
    f"base-test"
)
base.batch_size = 64
base.setup()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy(
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def to_tfrecord(x, y):
    feature = {
        "x": _float_feature(x.numpy().flatten()),  # one image in the list
        "y": _int64_feature(y.numpy().flatten()),  # one class in the list   
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)
                           ).SerializeToString()


def serialize(x, y):
    res = tf.py_function(to_tfrecord, (x, y), tf.string)
    return tf.reshape(res, ())

def deserialize(ex):
    features = {
        "x": tf.io.FixedLenFeature([128,128,1], tf.float32),
        "y": tf.io.FixedLenFeature([128,128,1], tf.int64),
    }
    return tf.io.parse_single_example(ex, features)

# final = base._train_ds.unbatch().map(serialize)
# writer = tf.data.experimental.TFRecordWriter("./res.tfrecord")
# writer.write(final)
ds = tf.data.TFRecordDataset(
    "./res.tfrecord", num_parallel_reads=tf.data.experimental.AUTOTUNE
).map(deserialize)
# for x,y in base._train_ds.unbatch():
#     print(x,y)
# for x,y in ds.map(lambda x: (x["x"], x["y"])):
#     print(x,y)