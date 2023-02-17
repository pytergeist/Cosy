import numpy as np
from cosy.models import CosyNet

import tensorflow as tf


input_ = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(input_)
x = tf.keras.layers.Dense(10)(x)
x = tf.keras.layers.Dense(10)(x)
x = tf.keras.layers.Dense(10)(x)
out = tf.keras.layers.Dense(1)(x)
test_model = tf.keras.Model(inputs=input_, outputs=out)


class CosyNetTest(tf.test.TestCase):
    def setUp(self):
        super(CosyNetTest, self).setUp()
        self.cosy_net = CosyNet(
            model_config=test_model.get_config(), number_models=3, layer_cutoff=-1
        )

    def tearDown(self):
        pass

    def test__build_task_models(self):
        model_list = self.cosy_net._build_task_models()
        assert isinstance(model_list, list)
        for model in model_list:
            assert isinstance(model, tf.keras.Model)

    def test__get_parameters(self):
        parameters = self.cosy_net._get_parameters()
        assert isinstance(parameters, list)
        assert len(parameters[0]) == self.cosy_net.number_models

        all_dense = [
            layer
            for layer in self.cosy_net.get_models()[0].layers
            if "dense" in layer.name  # change to be generic
        ]
        assert len(parameters) == len(all_dense) - 1

    def test_soft_loss(self):
        soft_loss = self.cosy_net.soft_loss()
        assert isinstance(soft_loss, tf.Tensor)
        assert 1e-12 < soft_loss < np.inf

    def test_get_models(self):
        models = self.cosy_net.get_models()
        assert isinstance(models, list)
        assert len(models) == self.cosy_net.number_models
        for model in models:
            assert isinstance(model, tf.keras.Model)

    def test_get_multi_weights(self):
        weights = self.cosy_net.get_multi_weights()
        assert isinstance(weights, list)
        assert len(weights) == self.cosy_net.number_models


if __name__ == "__main__":
    tf.test.main()
