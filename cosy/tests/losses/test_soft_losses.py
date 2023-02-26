import numpy as np
import tensorflow as tf
import pytest
from cosy.losses import (
    squared_frobenius_norm,
    trace_norm,
    l1_norm,
    kl_divergence,
    l2_norm,
)


@pytest.mark.parametrize(
    "parameters, lambdas, expected_loss",
    [
        ("network_params_ones", [0.5, 0.5, 0.5], 0.0),
        ("network_params_zeros", [0.5, 0.5, 0.5], 0.0),
        ("network_params_alternate", [0.5, 0.5, 0.5], 8.0),
        ("network_params_random_deterministic", [0.5, 0.5, 0.5], 2.131),
    ],
)
def test_squared_frobenius_norm(parameters, lambdas, expected_loss, request):
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
        ("network_params_random_deterministic", [0.5, 0.5, 0.5], 2.131),
    ],
)
def test_trace_norm(parameters, lambdas, expected_loss, request):
    params = request.getfixturevalue(parameters)
    loss = trace_norm(params, lambdas)
    assert tf.is_tensor(loss)
    assert loss.shape == []
    assert np.isclose(loss, expected_loss)


@pytest.mark.parametrize(
    "parameters, lambdas, expected_loss",
    [
        ("network_params_ones", [0.5, 0.5, 0.5], 0.0),
        ("network_params_zeros", [0.5, 0.5, 0.5], 0.0),
        ("network_params_alternate", [0.5, 0.5, 0.5], 4.0),
        ("network_params_random_deterministic", [0.5, 0.5, 0.5], 3.27016),
    ],
)
def test_l1_norm(parameters, lambdas, expected_loss, request):
    params = request.getfixturevalue(parameters)
    loss = l1_norm(params, lambdas)
    assert tf.is_tensor(loss)
    assert loss.shape == []
    assert np.isclose(loss, expected_loss)


@pytest.mark.parametrize(
    "parameters, lambdas, expected_loss",
    [
        ("network_params_ones", [0.5, 0.5, 0.5], 0.0),
        ("network_params_zeros", [0.5, 0.5, 0.5], 0.0),
        ("network_params_alternate", [0.5, 0.5, 0.5], 0.0),
        ("network_params_random_deterministic", [0.5, 0.5, 0.5], 0.28827),
    ],
)
def test_kl_divergence(parameters, lambdas, expected_loss, request):
    params = request.getfixturevalue(parameters)
    loss = kl_divergence(params, lambdas)
    assert tf.is_tensor(loss)
    assert loss.shape == []
    assert np.isclose(loss, expected_loss)


@pytest.mark.parametrize(
    "parameters, lambdas, expected_loss",
    [
        ("network_params_ones", [0.5, 0.5, 0.5], 0.0),
        ("network_params_zeros", [0.5, 0.5, 0.5], 0.0),
        ("network_params_alternate", [0.5, 0.5, 0.5], 4.0),
        ("network_params_random_deterministic", [0.5, 0.5, 0.5], 2.762555),
    ],
)
def test_l2_norm(parameters, lambdas, expected_loss, request):
    params = request.getfixturevalue(parameters)
    loss = l2_norm(params, lambdas)
    assert tf.is_tensor(loss)
    assert loss.shape == []
    assert np.isclose(loss, expected_loss)
