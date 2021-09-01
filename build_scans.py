from mpunet.models import UNet3D, UNet
from heartnet.models.base import BaseModelTraining
from tensorflow import keras
from heartnet.layers import RobustScaler
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
                             f"base-data-robust-{i}")
    base.batch_size = 32
    base.setup(True)
    # base.train()
    base.save_output("final")
    # base = BaseModelTraining(
    #     UNet(2, depth=4, dim=128, out_activation="softmax", complexity_factor=2),
    #     f"base{i}"
    # )
    # base.batch_size = 16
    # base.setup(True)
    # base.save_output("train")
    # base.save_output("val")
    # base.save_output("test")
    # base.save_output("final")
    # base3D = BaseModelTraining(
    #     UNet3D(2, dim=112, out_activation="softmax"), name=f"pad{i}"
    # )
    # base3D.batch_size = 1
    # base3D.setup(True)
    # base3D.save_output("train")
    # base3D.save_output("val")
    # base3D.save_output("test")
    # base3D.save_output("final")