import numpy as np
import tensorflow as tf
import pytest
from cosy.losses import (
    squared_frobenius_norm,
    trace_norm,
    MaskedMeanSquaredError,
    pairwise_loss_squared_frobenius,
    pairwise_loss_trace_norm,
)


@pytest.mark.parametrize(
    "parameters, lambdas, expected_loss",
    [
        ("network_params_ones", [0.5, 0.5, 0.5], 0.0),
        ("network_params_zeros", [0.5, 0.5, 0.5], 0.0),
        ("network_params_alternate", [0.5, 0.5, 0.5], 8.0),
    ],
)
def test_squared_frobenius_norm_with_params(
    parameters, lambdas, expected_loss, request
):
    params = request.getfixturevalue(parameters)
    loss = squared_frobenius_norm(params, lambdas)
    assert tf.is_tensor(loss)
    assert loss.shape == []
    assert np.isclose(loss, expected_loss)


@pytest.mark.parametrize(
    "parameters, lambdas, expected_loss",
    [
        ("network_params_ones", [0.5, 0.5, 0.5], 0.0),
        ("network_params_zeros", [0.5, 0.5, 0.5], 0.0),
        ("network_params_alternate", [0.5, 0.5, 0.5], 8.0),
    ],
)
def test_trace_norm_with_params(parameters, lambdas, expected_loss, request):
    params = request.getfixturevalue(parameters)
    loss = trace_norm(params, lambdas)
    assert tf.is_tensor(loss)
    assert loss.shape == []
    assert np.isclose(loss, expected_loss)


@pytest.mark.parametrize(
    "W, lambd, expected_loss",
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
def test_pairwise_loss_squared_frobenius(W, lambd, expected_loss):
    W_tf = [tf.constant(Wi, dtype=tf.float32) for Wi in W]
    expected_loss_tf = tf.constant(expected_loss, dtype=tf.float32)
    actual_loss_tf = pairwise_loss_squared_frobenius(W_tf, lambd)
    assert np.isclose(actual_loss_tf, expected_loss_tf, rtol=1e-5)


@pytest.mark.parametrize(
    "W, lambd, expected_loss",
    [
        (
            [
                np.array([[0.1, 2], [3, 4]]),
                np.array([[-0.5, 6], [7, -0.8]]),
                np.array([[11, 6], [2, 7]]),
            ],
            1.0,
            252,
        ),
        ([np.array([[1, 2], [3, 4]]), np.array([[0.1, 2], [0.3, 4]])], 0.5, 0.0),
        ([np.array([[1, 2], [3, 0.4]]), np.array([[2, 4], [6, 8]])], 2.0, 60.0),
    ],
)
def test_pairwise_loss_squared_frobenius(W, lambd, expected_loss):
    W_tf = [tf.constant(Wi, dtype=tf.float32) for Wi in W]
    expected_loss_tf = tf.constant(expected_loss, dtype=tf.float32)
    actual_loss_tf = pairwise_loss_trace_norm(W_tf, lambd)
    assert np.isclose(actual_loss_tf, expected_loss_tf, rtol=1e-5)


if __name__ == "__main__":
    pytest.main()
