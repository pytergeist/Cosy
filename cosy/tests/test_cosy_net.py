import numpy as np
import pytest
from cosy.models import CosyNet, CosyNetMultiInput
import tensorflow as tf


@pytest.mark.parametrize(
    "params",
    [
        "cosy_net_params",
        "cosy_net_params_multi_input",
    ],
)
def test_init_cosy_net(params, request):
    params = request.getfixturevalue(params)
    cosy_net = CosyNet(**params)
    assert isinstance(cosy_net, CosyNet)


def test__build_task_models(request):
    cosy_net = request.getfixturevalue("cosy_net_obj")
    model_list = cosy_net._build_task_models()
    assert isinstance(model_list, list)
    for model in model_list:
        assert isinstance(model, tf.keras.Model)


def test__get_parameters(request):
    cosy_net = request.getfixturevalue("cosy_net_obj")
    parameters = cosy_net._get_parameters()
    assert isinstance(parameters, list)
    assert len(parameters[0]) == cosy_net.number_models

    all_weights = [
        layer.weights
        for layer in cosy_net.get_models()[0].layers
        if any("kernel" in w.name for w in layer.weights)
    ]

    assert len(parameters) == len(all_weights) - 1


def test_soft_loss(request):
    cosy_net = request.getfixturevalue("cosy_net_obj")
    soft_loss = cosy_net.soft_loss()
    assert isinstance(soft_loss, tf.Tensor)
    assert 1e-12 <= soft_loss < np.inf


def test_get_models(request):
    cosy_net = request.getfixturevalue("cosy_net_obj")
    models = cosy_net.get_models()
    assert isinstance(models, list)
    assert len(models) == cosy_net.number_models
    for model in models:
        assert isinstance(model, tf.keras.Model)


def test_get_multi_weights(request):
    cosy_net = request.getfixturevalue("cosy_net_obj")
    weights = cosy_net.get_multi_weights()
    assert isinstance(weights, list)
    assert len(weights) == cosy_net.number_models


@pytest.mark.parametrize(
    "params",
    [
        "cosy_net_params",
        "cosy_net_params_multi_input",
    ],
)
def test_init_cosy_net_multi_input(params, request):
    params = request.getfixturevalue(params)
    cosy_net = CosyNetMultiInput(**params)
    assert isinstance(cosy_net, CosyNetMultiInput)


def test_cosy_net_multi_input__build_task_models(request):
    cosy_net = request.getfixturevalue("cosy_net_multi_input_obj")
    model_list = cosy_net._build_task_models()
    assert isinstance(model_list, list)
    for model in model_list:
        assert isinstance(model, tf.keras.Model)


if __name__ == "__main__":
    tf.test.main()
