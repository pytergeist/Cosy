import tensorflow as tf
from .pairwise_losses import (
    pairwise_loss_squared_frobenius,
    pairwise_loss_trace_norm,
    pairwise_loss_l1_norm,
    pairwise_loss_kl_divergence,
    pairwise_loss_wasserstein_distance,
)


def squared_frobenius_norm(parameters, lambdas):
    losses = []
    for W, lambd in zip(parameters, lambdas):
        losses.append(pairwise_loss_squared_frobenius(W, lambd))
    return tf.reduce_sum(losses)


def trace_norm(parameters, lambdas):
    losses = []
    for W, lambd in zip(parameters, lambdas):
        losses.append(pairwise_loss_trace_norm(W, lambd))
    return tf.reduce_sum(losses)


def l1_norm(parameters, lambdas):
    losses = []
    for W, lambd in zip(parameters, lambdas):
        losses.append(pairwise_loss_l1_norm(W, lambd))
    return tf.reduce_sum(losses)


def kl_divergence(parameters, lambdas):
    losses = []
    for W, lambd in zip(parameters, lambdas):
        losses.append(pairwise_loss_kl_divergence(W, lambd))
    return tf.reduce_sum(losses)


def wasserstein_distance(parameters, lambdas):
    losses = []
    for W, lambd in zip(parameters, lambdas):
        losses.append(pairwise_loss_wasserstein_distance(W, lambd))
    return tf.reduce_sum(losses)
