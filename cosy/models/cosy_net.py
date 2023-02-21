from typing import Dict, Callable, Union
from .base import BaseCosy
from cosy.losses import squared_frobenius_norm


class CosyNet(BaseCosy):
    def __init__(
        self,
        model_config: Dict,
        number_models: int,
        max_layer_cutoff: int = -1,
        min_layer_cutoff: int = 0,
        loss_fn: Callable = squared_frobenius_norm,
        scalar: Union[list, float] = 0.2,
        *args,
        **kwargs,
    ):
        super(CosyNet, self).__init__(
            model_config=model_config,
            number_models=number_models,
            max_layer_cutoff=max_layer_cutoff,
            min_layer_cutoff=min_layer_cutoff,
            loss_fn=loss_fn,
            scalar=scalar,
            *args,
            **kwargs,
        )

        self.task_nets = self._build_task_models()

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
        max_layer_cutoff: int = -1,
        min_layer_cutoff: int = 0,
        loss_fn: Callable = squared_frobenius_norm,
        scalar: float = 0.2,
        *args,
        **kwargs,
    ):
        super(CosyNetMultiInput, self).__init__(  # pragma: no cover
            model_config=model_config,
            number_models=number_models,
            max_layer_cutoff=max_layer_cutoff,
            min_layer_cutoff=min_layer_cutoff,
            loss_fn=loss_fn,
            scalar=scalar,
            *args,
            **kwargs,
        )

        self.task_nets = self._build_task_models()

    def call(self, inputs):
        soft_sharing_loss = self.soft_loss()
        [net.add_loss(soft_sharing_loss) for net in self.task_nets]
        self.add_metric(soft_sharing_loss, name="scaled_soft_loss", aggregation="mean")
        return tuple(task_net(inputs[i]) for i, task_net in enumerate(self.task_nets))
