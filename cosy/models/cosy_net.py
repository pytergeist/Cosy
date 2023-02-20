from typing import Dict, Callable
import tensorflow as tf
from .base import BaseCosy
from cosy.losses import l2_loss


class CosyNet(BaseCosy):
    def __init__(
        self,
        model_config: Dict,
        number_models: int,
        layer_cutoff: int = -1,
        loss_fn: Callable = l2_loss,
        scalar: float = 0.2,
        *args,
        **kwargs,
    ):
        super(CosyNet, self).__init__(
            model_config=model_config,
            number_models=number_models,
            layer_cutoff=layer_cutoff,
            loss_fn=loss_fn,
            scalar=scalar,
            *args,
            **kwargs,
        )

        self.task_nets = self._build_task_models()

    def _get_parameters(self):
        parameters = []
        for params in zip(*[layer.weights for layer in self.task_nets.layers]):
            if "kernel" in params[1].name:  # CHANGE THIS TO ALL STATEMENT
                parameters.append(params)

        return parameters[: self.layer_cutoff]

    def call(self, inputs):
        soft_sharing_loss = self.soft_loss()
        [net.add_loss(soft_sharing_loss) for net in self.task_nets]
        self.add_metric(soft_sharing_loss, name="scaled_soft_loss", aggregation="mean")
        return tuple(task_net(inputs) for task_net in self.task_nets)


class CosyNetMultiInput(BaseCosy):
    def __init__(
        self,
        model_config: Dict,
        number_models: int,
        layer_cutoff: int = -1,
        loss_fn: Callable = l2_loss,
        scalar: float = 0.2,
        *args,
        **kwargs,
    ):
        super(CosyNetMultiInput, self).__init__(
            model_config=model_config,
            number_models=number_models,
            layer_cutoff=layer_cutoff,
            loss_fn=loss_fn,
            scalar=scalar,
            *args,
            **kwargs,
        )

        self.task_nets = self._build_task_models()

    def _get_parameters(self):
        parameters = []
        for idx, params in enumerate(
            zip(*[layer.weights for layer in self.task_nets.layers])
        ):
            if "kernel" in params[1].name and idx > 0:
                parameters.append(params)

        return parameters[: self.layer_cutoff]

    def call(self, inputs):
        soft_sharing_loss = self.soft_loss()
        [net.add_loss(soft_sharing_loss) for net in self.task_nets]
        self.add_metric(soft_sharing_loss, name="scaled_soft_loss", aggregation="mean")
        return tuple(task_net(inputs[i]) for i, task_net in enumerate(self.task_nets))

    def summary(self):
        x = tf.keras.Input(shape=(24, 24, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
