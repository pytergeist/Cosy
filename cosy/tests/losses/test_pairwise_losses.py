import numpy as np
import tensorflow as tf
import pytest
from cosy.losses import (
    pairwise_loss_squared_frobenius,
    pairwise_loss_trace_norm,
)


@pytest.mark.parametrize(
    "weights, lambd, expected_loss",
    [
        (
            [
                np.array([[1, 2], [3, 4]]),
                np.array([[5, 6], [7, 8]]),
                np.array([[11, 6], [2, 7]]),
            ],
            1.0,
            252,
        ),
        ([np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])], 0.5, 0.0),
        ([np.array([[1, 2], [3, 4]]), np.array([[2, 4], [6, 8]])], 2.0, 60.0),
    ],
)
def test_pairwise_loss_squared_frobenius(weights, lambd, expected_loss):
    W_tf = [tf.constant(Wi, dtype=tf.float32) for Wi in weights]
    expected_loss_tf = tf.constant(expected_loss, dtype=tf.float32)
    actual_loss_tf = pairwise_loss_squared_frobenius(W_tf, lambd)
    assert np.isclose(actual_loss_tf, expected_loss_tf, rtol=1e-5)


@pytest.mark.parametrize(
    "weights, lambd, expected_loss",
    [
        (
            [
                np.array([[1, 2], [3, 4]]),
                np.array([[5, 6], [7, 8]]),
                np.array([[11, 6], [2, 7]]),
            ],
            1.0,
            252,
        ),
        ([np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])], 0.5, 0.0),
        ([np.array([[1, 2], [3, 4]]), np.array([[2, 4], [6, 8]])], 2.0, 60.0),
    ],
)
def test_pairwise_loss_squared_frobenius(weights, lambd, expected_loss):
    W_tf = [tf.constant(Wi, dtype=tf.float32) for Wi in weights]
    expected_loss_tf = tf.constant(expected_loss, dtype=tf.float32)
    actual_loss_tf = pairwise_loss_trace_norm(W_tf, lambd)
    assert np.isclose(actual_loss_tf, expected_loss_tf, rtol=1e-5)


if __name__ == "__main__":
    pytest.main()
