import numpy as np
import tensorflow as tf
import pytest
from cosy.losses import (
    pairwise_loss_squared_frobenius,
    pairwise_loss_trace_norm,
    pairwise_loss_l1_norm,
    pairwise_loss_kl_divergence,
    pairwise_loss_wasserstein_distance,
)


@pytest.mark.parametrize(
    "weights, lambdas, expected_loss",
    [
        (
            'weights_2d',
            [1.0],
            16.0,
        ),
        ('weights_2d_identical', [0.5], 0.0),
        (
            'weights_3d',
            [1.0, 0.5, 1.0],
            64.0,
        ),
        (
            'weights_4d',
            [1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
            240,
        ),
    ],
)
def test_pairwise_loss_squared_frobenius(weights, lambdas, expected_loss, request):
    weights = request.getfixturevalue(weights)
    W_tf = [tf.constant(Wi, dtype=tf.float32) for Wi in weights]
    expected_loss_tf = tf.constant(expected_loss, dtype=tf.float32)
    actual_loss_tf = pairwise_loss_squared_frobenius(W_tf, lambdas)
    assert np.isclose(actual_loss_tf, expected_loss_tf, rtol=1e-5)


@pytest.mark.parametrize(
    "weights, lambdas, expected_loss",
    [
        (
            'weights_2d',
            [1.0],
            16.0,
        ),
        ('weights_2d_identical', [0.5], 0.0),
        (
            'weights_3d',
            [1.0, 0.5, 1.0],
            64.0,
        ),
        (
            'weights_4d',
            [1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
            240,
        ),
    ],
)
def test_pairwise_loss_trace_norm(weights, lambdas, expected_loss, request):
    weights = request.getfixturevalue(weights)
    W_tf = [tf.constant(Wi, dtype=tf.float32) for Wi in weights]
    expected_loss_tf = tf.constant(expected_loss, dtype=tf.float32)
    actual_loss_tf = pairwise_loss_trace_norm(W_tf, lambdas)
    assert np.isclose(actual_loss_tf, expected_loss_tf, rtol=1e-5)


@pytest.mark.parametrize(
    "weights, lambdas, expected_loss",
    [
        (
            'weights_2d',
            [1.0],
            4.0,
        ),
        ('weights_2d_identical', [0.5], 0.0),
        (
            'weights_3d',
            [1.0, 0.5, 1.0],
            12.0,
        ),
        (
            'weights_4d',
            [1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
            32.0,
        ),
    ],
)
def test_pairwise_loss_l1_norm(weights, lambdas, expected_loss, request):
    weights = request.getfixturevalue(weights)
    W_tf = [tf.constant(Wi, dtype=tf.float32) for Wi in weights]
    expected_loss_tf = tf.constant(expected_loss, dtype=tf.float32)
    actual_loss_tf = pairwise_loss_l1_norm(W_tf, lambdas)
    assert np.isclose(actual_loss_tf, expected_loss_tf, rtol=1e-5)


@pytest.mark.parametrize(
    "weights, lambdas, expected_loss",
    [
        (
            np.array([[[0.2040, 0.5960], [0.27485, 0.9938]], [[0.4872, 0.375682], [0.389284, 0.99984]]]),
            [1.0],
            0.032499,
        ),
        (
                np.array([[[0.1, 0.1], [0.1, 0.1]], [[0.1, 0.1], [0.1, 0.1]]]),
                [0.5],
                0.0),
        (
            np.array([
                [[0.2040, 0.5960], [0.27485, 0.9938]],
                [[0.4872, 0.375682], [0.389284, 0.99984]],
                [[0.78929, 0.1203], [0.89473, 0.7892734]],
            ]),
            [1.0, 0.5, 1.0],
            0.24123,
        ),
        (
            np.array([
                [[0.2040, 0.5960], [0.27485, 0.9938]],
                [[0.4872, 0.375682], [0.389284, 0.99984]],
                [[0.78929, 0.1203], [0.89473, 0.7892734]],
                [[0.123, 0.456], [0.789, 0.123]],
            ]),
            [1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
            0.79705,
        ),
    ],
)
def test_pairwise_loss_kl_divergence(weights, lambdas, expected_loss):
    W_tf = [tf.constant(Wi, dtype=tf.float32) for Wi in weights]
    expected_loss_tf = tf.constant(expected_loss, dtype=tf.float32)
    actual_loss_tf = pairwise_loss_kl_divergence(W_tf, lambdas)
    assert np.isclose(actual_loss_tf, expected_loss_tf, rtol=1e-4)


@pytest.mark.parametrize(
    "weights, lambdas, expected_loss",
    [
        (
            'weights_2d',
            [1.0],
            4.0,
        ),
        ('weights_2d_identical', [0.5], 0.0),
        (
            'weights_3d',
            [1.0, 0.5, 1.0],
            12.0,
        ),
        (
            'weights_4d',
            [1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
            32.0,
        ),
    ],
)
def test_pairwise_loss_wasserstein_distance(weights, lambdas, expected_loss, request):
    weights = request.getfixturevalue(weights)
    W_tf = [tf.constant(Wi, dtype=tf.float32) for Wi in weights]
    expected_loss_tf = tf.constant(expected_loss, dtype=tf.float32)
    actual_loss_tf = pairwise_loss_wasserstein_distance(W_tf, lambdas)
    assert np.isclose(actual_loss_tf, expected_loss_tf, rtol=1e-5)


if __name__ == "__main__":
    pytest.main()
