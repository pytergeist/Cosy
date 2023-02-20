from abc import ABC, abstractmethod
from typing import Dict, Callable
import tensorflow as tf
import numpy as np

from cosy.losses import l2_loss


class BaseCosy(ABC, tf.keras.Model):
    def __init__(
        self,
        model_config: Dict,
        number_models: int,
        layer_cutoff: int = -1,
        loss_fn: Callable = l2_loss,
        scalar: float = 0.2,
    ):
        super(BaseCosy, self).__init__()

        if isinstance(model_config, list):
            self.model_config = model_config
        else:
            self.model_config = [model_config] * number_models

        self.number_models = number_models
        self.layer_cutoff = layer_cutoff
        self.loss_fn = loss_fn
        self.scalar = scalar

    def _build_task_models(self):
        task_nets = [tf.keras.Model.from_config(config) for config in self.model_config]
        for net in task_nets:
            net.compile()
        return task_nets

    @abstractmethod
    def _get_parameters(self) -> list:
        pass

    def soft_loss(self):
        parameters = self._get_parameters()
        soft_sharing_loss = self.scalar * tf.reduce_sum(self.loss_fn(parameters))
        return tf.keras.backend.clip(soft_sharing_loss, 1e-12, np.inf)

    @abstractmethod
    def call(self, inputs) -> tuple:
        pass

    def get_models(self):
        return [task_net for task_net in self.task_nets]

    def get_multi_weights(self):
        return [task_net.get_weights() for task_net in self.task_nets]
