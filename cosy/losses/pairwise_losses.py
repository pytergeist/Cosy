import tensorflow as tf
import itertools


def pairwise_loss_squared_frobenius(weights, lambdas):
    squared_norms = []
    for lmbd_idx, (Wi, Wj) in enumerate(itertools.combinations(weights, 2)):
        squared_norm = tf.norm(Wi - Wj, ord="fro", axis=(0, 1)) ** 2
        scaled_squared_norm = tf.gather(lambdas, lmbd_idx) * squared_norm
        squared_norms.append(scaled_squared_norm)
    return tf.reduce_sum(squared_norms)


def pairwise_loss_trace_norm(weights, lambdas):
    trace_norms = []
    for lmbd_idx, (Wi, Wj) in enumerate(itertools.combinations(weights, 2)):
        trace_norm = tf.linalg.trace(
            tf.matmul(tf.transpose(Wi - Wj), Wi - Wj)
        )
        scaled_trace_norm = tf.gather(lambdas, lmbd_idx) * trace_norm
        trace_norms.append(scaled_trace_norm)
    return tf.reduce_sum(trace_norms)


def pairwise_loss_l1_norm(weights, lambdas):
    l1_norms = []
    for lmbd_idx, (Wi, Wj) in enumerate(itertools.combinations(weights, 2)):
        l1_norm = tf.norm(Wi - Wj, ord=1, axis=(0, 1))
        scaled_l1_norm = tf.gather(lambdas, lmbd_idx) * l1_norm
        l1_norms.append(scaled_l1_norm)
    return tf.reduce_sum(l1_norms)


def pairwise_loss_kl_divergence(weights, lambdas):
    kl_divergnces = []
    for lmbd_idx, (Wi, Wj) in enumerate(itertools.combinations(weights, 2)):
        kl_divergnce = tf.nn.softmax(Wi) * (
            tf.math.log(tf.nn.softmax(Wi)) - tf.math.log(tf.nn.softmax(Wj))
        )
        scaled_kl_divergnce = tf.gather(lambdas, lmbd_idx) * kl_divergnce
        kl_divergnces.append(scaled_kl_divergnce)
    return tf.reduce_sum(kl_divergnces)


def pairwise_loss_l2_norm(weights, lambdas):
    l2_norms = []
    for lmbd_idx, (Wi, Wj) in enumerate(itertools.combinations(weights, 2)):
        l2_norm = tf.norm(Wi - Wj, ord=2, axis=(0, 1))
        scaled_l2_norm = (
            tf.gather(lambdas, lmbd_idx) * l2_norm
        )
        l2_norms.append(scaled_l2_norm)
    return tf.reduce_sum(l2_norms)
