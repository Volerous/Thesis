from os import PathLike
from typing import Any, Callable, Dict

import numpy as np
import tensorflow as tf
import pathlib
import nibabel as nib
from .preprocess import *

SPLITS = {"train": 26, "val": 9, "test": 9, "true_test": 0}


def base_loader(base_dir: PathLike, **kwargs) -> tf.data.Dataset:
    base_dir = pathlib.Path(base_dir)
    images = (base_dir / "images")
    labels = (base_dir / "labels")
    img_ds = tf.data.Dataset.from_tensor_slices(
        [str(i) for i in images.glob("*")])
    label_ds = tf.data.Dataset.from_tensor_slices(
        [str(i) for i in labels.glob("*")])
    dataset = tf.data.Dataset.zip((img_ds, label_ds))

    def load_img(x, y):
        ret_x = nib.load(x.numpy().decode("utf-8")).get_fdata()
        ret_y = nib.load(y.numpy().decode("utf-8")).get_fdata()
        if kwargs.get("augmentations", []):
            for aug in kwargs.get("augmentations", []):
                ret_x, ret_y = aug(ret_x, ret_y)
        return np.squeeze(ret_x), np.squeeze(ret_y)

    def get_img(x, y):
        return tf.py_function(load_img, [x, y], [tf.float32, tf.int32])

    dataset = dataset.map(get_img,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def load3D(base_dir, output_dim=111, **kwargs):
    dataset = base_loader(base_dir, **kwargs)
    dataset = dataset.map(crop_image_to_shape(output_dim), -1)
    dataset = dataset.map(expand_dims, -1)
    dataset = dataset.map(crop_image_to_shape(output_dim), -1)
    return dataset


def load2D(base_dir, **kwargs):
    dataset = base_loader(base_dir, **kwargs)
    dataset = dataset.map(transpose_slices, -1)
    dataset = dataset.map(expand_dims, -1)
    return dataset.flat_map(split_slices)


load_functions: Dict[str, Callable[[Any], tf.data.Dataset]] = {
    "UNet": load2D,
    "UNet3D": load3D
}


def deserialize(ex):
    features = {
        "x": tf.io.FixedLenFeature([128, 128, 1], tf.float32),
        "y": tf.io.FixedLenFeature([128, 128, 1], tf.int64),
    }
    return tf.io.parse_single_example(ex, features)


def test_load(base_dir, **kwargs):
    return tf.data.TFRecordDataset(
        base_dir,
        num_parallel_reads=-1).map(deserialize).map(lambda x: (x["x"], x["y"]))


# def test_load()