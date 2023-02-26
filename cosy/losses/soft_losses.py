import tensorflow as tf
from .pairwise_losses import (
    pairwise_loss_squared_frobenius,
    pairwise_loss_trace_norm,
    pairwise_loss_l1_norm,
    pairwise_loss_kl_divergence,
    pairwise_loss_l2_norm,
)


def squared_frobenius_norm(parameters, lambdas):
    losses = []
    for W in parameters:
        losses.append(pairwise_loss_squared_frobenius(W, lambdas))
    return tf.reduce_sum(losses)


def trace_norm(parameters, lambdas):
    losses = []
    for W in parameters:
        losses.append(pairwise_loss_trace_norm(W, lambdas))
    return tf.reduce_sum(losses)


def l1_norm(parameters, lambdas):
    losses = []
    for W in parameters:
        losses.append(pairwise_loss_l1_norm(W, lambdas))
    return tf.reduce_sum(losses)


def kl_divergence(parameters, lambdas):
    losses = []
    for W in parameters:
        losses.append(pairwise_loss_kl_divergence(W, lambdas))
    return tf.reduce_sum(losses)


def l2_norm(parameters, lambdas):
    losses = []
    for W in parameters:
        losses.append(pairwise_loss_l2_norm(W, lambdas))
    return tf.reduce_sum(losses)
