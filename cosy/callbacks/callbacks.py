import tensorflow as tf
import numpy as np
from typing import List


class EarlyStoppingMultiLoss(tf.keras.callbacks.Callback):
    """Stop training when multiple sopping criteria have been met.

    Arguments:
        monitor: List of strings. The metrics to monitor.
        patience: Number of epochs to wait after stopping criteria has been hit. After this
        number of no improvement in every criterion, training stops.
    """

    def __init__(self, monitor: List[str], patience: int = 0, mode: str = "min"):
        super(EarlyStoppingMultiLoss, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best_weights = [None] * len(self.monitor)

    def on_train_begin(self, logs=None):
        self.wait = [0] * len(self.monitor)
        self.stopped_epoch = 0
        if self.mode == "min":
            self.best_multi = [np.inf] * len(self.monitor)
            self.inequality = np.less
        elif self.mode == "max":
            self.best_multi = [-1 * np.inf] * len(self.monitor)
            self.inequality = np.greater

    def on_epoch_end(self, epoch, logs=None):
        for idx, loss_val in enumerate(self.monitor):
            current = logs.get(loss_val)
            if self.inequality(current, self.best_multi[idx]):
                self.best_multi[idx] = current
                self.wait[idx] = 0
                # Record the best weights if current results is better (less).
                self.best_weights[idx] = self.model.get_multi_weights()[idx]

            else:
                self.wait[idx] += 1

                if all(wait_value >= self.patience for wait_value in self.wait):
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    best_weights = []
                    for weight in self.best_weights:
                        best_weights += weight
                    self.model.set_weights(best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
