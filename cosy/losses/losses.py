import tensorflow as tf
from functools import reduce


class L2Loss(tf.keras.losses.Loss):
    def call(self, params):
        reduced_params = reduce(tf.math.subtract, params)
        return tf.norm(reduced_params, ord="fro", axis=(0, 1))


def l2_loss(params):
    reduced_params = reduce(tf.math.subtract, params)
    return tf.norm(reduced_params, ord="fro", axis=(0, 1))
