from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import Callback


class CSVEvaluateLogger(CSVLogger):

    def __init__(self, filename, separator=',', append=False):
        super().__init__(filename, separator=separator, append=append)

    def on_test_begin(self, logs):
        super().on_train_begin(logs=logs)

    def on_test_end(self, logs):
        super().on_epoch_end(0, logs=logs)
        super().on_train_end(logs=logs)


class DicePerPerson(Callback):

    def __init__(self):
        super().__init__()
        self.genders = [
            "M", "M", "F", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M",
            "F", "F", "F", "M", "F", "M", "M", "F", "F", "F", "M", "M", "M",
            "M", "M", "F", "F", "F", "M", "M", "M", "M", "M", "M", "M", "F",
            "F", "M", "M", "F", "M", "M", "F", "M", "F", "M", "M", "M", "M",
            "M", "F", "M", "F", "M", "F", "F", "M", "M"
        ]
        self.idx = 44
        self.start_idx = 44
        self.men = 0
        self.women = 0

    def on_test_batch_end(self, batch, logs):
        if self.genders[self.idx] == "M":
            self.men += logs["dice"]
        elif self.genders[self.idx] == "F":
            self.women += logs["dice"]
        self.idx += 1

    def on_test_end(self, logs):
        logs["dice_men"] = self.men / len(
            [i for i in self.genders[self.start_idx:self.idx] if i == "M"]
        )
        logs["dice_women"] = self.women / len(
            [i for i in self.genders[self.start_idx:self.idx] if i == "F"]
        )
        print(logs)