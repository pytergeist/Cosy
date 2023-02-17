from typing import Dict, Callable
import tensorflow as tf
import numpy as np

from cosy.losses import l2_loss


class CosyNet(tf.keras.Model):
    def __init__(
        self,
        model_config: Dict,
        number_models: int,
        layer_cutoff: int = -1,
        loss_fn: Callable = l2_loss,
        scalar: float = 0.2,
    ):
        super(CosyNet, self).__init__()

        self.model_config = model_config
        self.number_models = number_models
        self.layer_cutoff = layer_cutoff
        self.loss_fn = loss_fn
        self.scalar = scalar
        self.task_nets = self._build_task_models()

    def _build_task_models(self):
        return [
            tf.keras.Model.from_config(self.model_config)
            for _ in range(self.number_models)
        ]

    def _get_parameters(self):
        parameters = []
        for params in zip(*[layer.weights for layer in self.task_nets.layers]):
            if "kernel" in params[1].name:
                parameters.append(params)

        return parameters[: self.layer_cutoff]

    def soft_loss(self):
        parameters = self._get_parameters()
        losses = []

        for params in parameters:
            losses.append(self.loss_fn(params))

        soft_sharing_loss = self.scalar * tf.reduce_sum(losses)
        return tf.keras.backend.clip(soft_sharing_loss, 1e-12, np.inf)

    def call(self, x):
        soft_sharing_loss = tf.constant(self.soft_loss())
        [net.add_loss(lambda: soft_sharing_loss) for net in self.task_nets]

        scaled_soft_loss = tf.identity(soft_sharing_loss, name="scaled_soft_loss")

        self.add_metric(scaled_soft_loss, name="scaled_soft_loss", aggregation="mean")

        return tuple(task_net(x) for task_net in self.task_nets)

    def get_models(self):
        return [task_net for task_net in self.task_nets]

    def get_multi_weights(self):
        return [task_net.get_weights() for task_net in self.task_nets]
