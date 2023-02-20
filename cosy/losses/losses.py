import tensorflow as tf
from functools import reduce


def l2_loss(parameters):
    loss = tf.constant(0.0)
    for params in parameters:
        reduced_param = reduce(tf.math.subtract, params)
        loss += tf.norm(reduced_param, ord="fro", axis=(0, 1))
    return loss


def trace_norm_loss(parameters):
    loss = tf.constant(0.0)
    for params in parameters:
        reduced_param = reduce(tf.math.subtract, params)
        singular_values = tf.linalg.svd(reduced_param, compute_uv=False)
        loss += tf.reduce_sum(singular_values)
    return loss


class MaskedMeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        mask = tf.not_equal(y_true, -1)
        y_pred_masked = tf.expand_dims(tf.boolean_mask(y_pred, mask), -1)
        y_true_masked = tf.expand_dims(tf.boolean_mask(y_true, mask), -1)

        return tf.keras.backend.mean(
            tf.math.squared_difference(y_pred_masked, y_true_masked), axis=-1
        )
