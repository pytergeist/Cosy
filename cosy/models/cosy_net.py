from typing import Dict, Callable
import tensorflow as tf
import numpy as np

from .task_model import TaskModel
from cosy.losses import l2_loss


class CosyNet(tf.keras.Model):
    def __init__(
        self,
        model_config: Dict,
        output_shape: int,
        model_strategy: str = "mlp",
        layer_cutoff: int = -1,
        loss_fn: Callable = l2_loss,
        scalar: float = 0.2,

    ):
        super(CosyNet, self).__init__()

        self.model_config = model_config
        self.output_shape = output_shape
        self.model_strategy = model_strategy
        self.layer_cutoff = layer_cutoff
        self.loss_fn = loss_fn
        self.scalar = scalar

    def _build_task_models(self):
        self.task_nets = [
            TaskModel.build_model_from_config(self.model_config)
            for _ in range(self.output_shape)
        ]

    def _get_parameters(self):
        parameters = []
        for params in zip(*[layer.weights for layer in self.task_nets.layers]):
            if "kernel" in params[1].name:
                parameters.append(params)

        return parameters[:self.layer_cutoff]

    def soft_loss(self):
        parameters = self._get_parameters()
        soft_sharing_loss = tf.constant(0.0)

        for params in parameters:
            soft_sharing_loss += self.loss_fn(params)

        return self.scalar * tf.keras.backend.clip(soft_sharing_loss, 1e-12, np.inf)

    def call(self, x):
        soft_sharing_loss = self.soft_loss()
        [net.add_loss(soft_sharing_loss) for net in self.task_nets]

        self.add_metric(soft_sharing_loss, name="scaled_soft_loss", aggregation="mean")

        return tuple(task_net(x) for task_net in self.task_nets)

    def get_models(self):
        return [task_net for task_net in self.task_nets]

    def get_multi_weights(self):
        return [task_net.get_weights() for task_net in self.task_nets]

    def test_build(self):
        x = tf.keras.Input(shape=(64, 345))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
