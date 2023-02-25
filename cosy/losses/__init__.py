from .soft_losses import (
    squared_frobenius_norm,
    l1_norm,
    trace_norm,
    kl_divergence,
    wasserstein_distance,
)
from .masked_losses import MaskedMeanSquaredError

from .soft_losses import (
    pairwise_loss_trace_norm,
    pairwise_loss_squared_frobenius,
    pairwise_loss_wasserstein_distance,
    pairwise_loss_kl_divergence,
    pairwise_loss_l1_norm,
)

__all__ = [
    "squared_frobenius_norm",
    "trace_norm",
    "l1_norm",
    "kl_divergence",
    "wasserstein_distance",
    "MaskedMeanSquaredError",
    "pairwise_loss_squared_frobenius",
    "pairwise_loss_trace_norm",
]
