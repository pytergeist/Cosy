import tensorflow as tf
import itertools


def pairwise_loss_squared_frobenius(W, lambd):
    squared_norms = []
    for Wi, Wj in itertools.combinations(W, 2):
        squared_norms.append(tf.norm(Wi - Wj, ord="fro", axis=(0, 1)) ** 2)
    loss = lambd * tf.reduce_sum(squared_norms)
    return loss


def pairwise_loss_trace_norm(W, lambd):
    loss = 0.0
    for Wi, Wj in itertools.combinations(W, 2):
        loss += tf.linalg.trace(
            tf.matmul(tf.transpose(tf.math.subtract(Wi, Wj)), tf.math.subtract(Wi, Wj))
        )
    return lambd * loss


def pairwise_loss_l1_norm(W, lambd):
    loss = 0.0
    for Wi, Wj in itertools.combinations(W, 2):
        loss += tf.norm(Wi - Wj, ord=1, axis=(0, 1))
    return lambd * loss


def pairwise_loss_kl_divergence(W, lambd):
    loss = 0.0
    for Wi, Wj in itertools.combinations(W, 2):
        loss += tf.reduce_sum(
            tf.nn.softmax(Wi)
            * (tf.math.log(tf.nn.softmax(Wi)) - tf.math.log(tf.nn.softmax(Wj)))
        )
    return lambd * loss


# def pairwise_loss_mutual_information(W, lambd):
#     loss = 0.0
#     for Wi, Wj in itertools.combinations(W, 2):
#         p = tf.nn.softmax(Wi)
#         q = tf.nn.softmax(Wj)
#         loss += tf.reduce_sum(
#             p * (tf.math.log(p) - tf.math.log(tf.reduce_sum(q, axis=0)))
#         ) + tf.reduce_sum(q * (tf.math.log(q) - tf.math.log(tf.reduce_sum(p, axis=0))))
#     return lambd * loss


def pairwise_loss_wasserstein_distance(W, lambd):
    loss = 0.0
    for Wi, Wj in itertools.combinations(W, 2):
        loss += tf.norm(Wi - Wj, ord=2, axis=(0, 1))
    return lambd * loss
