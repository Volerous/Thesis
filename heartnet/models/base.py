from typing import List
from mpunet.models.unet import UNet

from numpy.lib.function_base import append
from tensorflow import keras
from tensorflow.keras.metrics import Precision, Recall
from heartnet.callbacks.base import CSVEvaluateLogger, DicePerPerson
from heartnet.loader.base_loader import *
from ..metrics.classes import *
from tensorflow.keras import *
import nibabel as nib


class BaseModelTraining(object):

    def __init__(
        self, model, name, loss=None, augmentations=[], full=False
    ) -> None:
        super().__init__()
        self.name = name
        self.model: models.Model = model
        # self.num_slices = 111
        # self.image_size = self.model.img_shape
        # self.dim = self.image_size[0]
        if isinstance(model, keras.Sequential):
            self.core_model = model.layers[-1]
            self.model = keras.Model(inputs=model.inputs, outputs=model.outputs)
        else:
            self.core_model = model
        self.data_train_folder = "/homes/pmcd/Peter_Patrick3/train"
        self.data_val_folder = "/homes/pmcd/Peter_Patrick3/val"
        self.data_test_folder = "/homes/pmcd/Peter_Patrick3/test"
        self.data_final_test_folder = "/homes/pmcd/Peter_Patrick3/true_test"
        self.batch_size = 1
        self.metrics = [Dice(), FGF1Score(), FGRecall(), FGPrecision()]
        self.loss = loss or losses.SparseCategoricalCrossentropy()
        self.epochs = 500
        mult = self.batch_size / 16 if self.batch_size > 16 else 1
        self.optimizer = optimizers.Adam(
            1e-4 * (mult), 0.9, 0.999, 1e-8, decay=0.0
        )
        self.augmentations = augmentations
        self.aug_repeats = 0
        self.concat_augs = False
        self.final = full
        self._file_name = f"{self.model_name}_{self.name}"
        self.callbacks = [
            callbacks.CSVLogger(f"./logs/{self._file_name}.csv"),
            callbacks.ReduceLROnPlateau(
                patience=2,
                factor=0.90,
                verbose=1,
                monitor="val_dice",
                mode="max"
            ),
            callbacks.EarlyStopping(
                monitor='val_dice',
                min_delta=0,
                patience=11,
                verbose=1,
                mode='max',
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                f"./model/{self._file_name}.h5",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
                monitor="val_dice",
                mode="max"
            ),
        ]

    @property
    def model_name(self):
        return self.core_model.__class__.__name__

    def setup(self, load_weights=False):
        if load_weights:
            self.load_weights()
        self._train_ds, self._val_ds, self._test_ds, self._final_ds = self.load_datasets(
        )

        if self.final:
            self._train_ds = self._train_ds.concatenate(self._val_ds)
            self._val_ds = self._test_ds
            self._test_ds = self._final_ds
        self.model.compile(self.optimizer, self.loss, metrics=self.metrics)

    def train(self):
        self.model.fit(
            self._train_ds,
            validation_data=self._val_ds,
            epochs=self.epochs,
            callbacks=self.callbacks,
        )

    def evaluate(self, final=False):
        cbs = [
            CSVEvaluateLogger(
                f"./logs/{self._file_name}{'-final' if self.final or final else ''}-evaluate.csv"
            ),
        ]
        ds = self._test_ds
        if final or self.final:
            ds = self._final_ds
        self.model.evaluate(ds, callbacks=cbs)

    def save_output(self, ds):
        if ds == "train":
            ds = self._train_ds
            start = 0
        elif ds == "val":
            ds = self._val_ds
            start = 26
        elif ds == "test":
            ds = self._test_ds
            start = 35
        elif ds == "final":
            ds = self._final_ds
            start = 44
        if isinstance(self.core_model, UNet):
            res = self.model.predict(ds.unbatch().batch(111))
            res = tf.argmax(res, axis=-1)
            res = tf.reshape(res, [-1, 111, 128, 128])
            for idx, out in enumerate(res, start=start):
                out = tf.transpose(out, [1, 2, 0])
                img = nib.Nifti1Image(out.numpy(), np.eye(4))
                nib.save(
                    img,
                    f"/homes/pmcd/Peter_Patrick3/out/{self._file_name}-anon{idx}.nii"
                )
            return
        res = self.model.predict(ds)
        res = tf.squeeze(res)
        res = tf.argmax(res, axis=-1)
        if res.shape[3] > 111:
            res = res[..., :-1]
        diff = 128 - res.shape[1]
        res = tf.pad(
            res,
            [[0, 0], [diff // 2, diff // 2], [diff // 2, diff // 2], [0, 0]]
        )
        for idx, i in enumerate(res, start=start):
            img = nib.Nifti1Image(i.numpy(), np.eye(4))
            nib.save(
                img,
                f"/homes/pmcd/Peter_Patrick3/out/{self._file_name}-anon{idx}.nii"
            )

    def load_weights(self):
        self.model.load_weights(f"./model/{self._file_name}.h5")

    def load_datasets(self) -> List[tf.data.Dataset]:
        load_function = load_functions[self.model_name]
        splits = [
            self.data_train_folder, self.data_val_folder, self.data_test_folder,
            self.data_final_test_folder
        ]
        ret = {i: None for i in splits}
        for split in splits:
            if split.endswith(".tfrecord"):
                ds = test_load(split)
            else:
                ds = load_function(
                    split,
                    output_dim=self.core_model.img_shape[0],
                    augmentations=self.augmentations if split == "train" else []
                )
            if self.batch_size:
                ds = ds.batch(self.batch_size)
            ret[split] = ds.prefetch(-1)
        return list(ret.values())