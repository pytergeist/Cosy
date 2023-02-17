import tensorflow as tf
from functools import reduce


def l2_loss(params):
    reduced_params = reduce(tf.math.subtract, params)
    return tf.norm(
                reduced_params, ord="fro", axis=(0, 1)
            )


