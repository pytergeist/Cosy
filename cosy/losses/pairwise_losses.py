import tensorflow as tf
import itertools


def pairwise_loss_squared_frobenius(weights, lambd):
    squared_norms = []
    for Wi, Wj in itertools.combinations(weights, 2):
        squared_norms.append(tf.norm(Wi - Wj, ord="fro", axis=(0, 1)) ** 2)
    loss = lambd * tf.reduce_sum(squared_norms)
    return loss


def pairwise_loss_trace_norm(weights, lambd):
    loss = 0.0
    for Wi, Wj in itertools.combinations(weights, 2):
        loss += tf.linalg.trace(
            tf.matmul(tf.transpose(tf.math.subtract(Wi, Wj)), tf.math.subtract(Wi, Wj))
        )
    return lambd * loss


def pairwise_loss_l1_norm(weights, lambd):
    loss = 0.0
    for Wi, Wj in itertools.combinations(weights, 2):
        loss += tf.norm(Wi - Wj, ord=1, axis=(0, 1))
    return lambd * loss


def pairwise_loss_kl_divergence(weights, lambd):
    loss = 0.0
    for Wi, Wj in itertools.combinations(weights, 2):
        loss += tf.reduce_sum(
            tf.nn.softmax(Wi)
            * (tf.math.log(tf.nn.softmax(Wi)) - tf.math.log(tf.nn.softmax(Wj)))
        )
    return lambd * loss


def pairwise_loss_wasserstein_distance(weights, lambd):
    loss = 0.0
    for Wi, Wj in itertools.combinations(weights, 2):
        loss += tf.norm(Wi - Wj, ord=2, axis=(0, 1))
    return lambd * loss
