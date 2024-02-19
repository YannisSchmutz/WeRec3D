import tensorflow as tf
import numpy as np

try:
    from helpers import calculate_covariance_matrix, reshape_batch_for_cm
except ImportError:
    from models.helpers import calculate_covariance_matrix, reshape_batch_for_cm


@tf.function
def loss_total(m_mae, mae, alpha=0.5):
    """

    :param m_mae:
    :param mae:
    :param alpha:
    :return:
    >>> loss_total(1, 1, 0.5)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
    >>> loss_total(1, 1, 0.25)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
    >>> loss_total(10, 8, 0.5)
    <tf.Tensor: shape=(), dtype=float32, numpy=9.0>
    >>> loss_total(10, 8, 0.25)
    <tf.Tensor: shape=(), dtype=float32, numpy=8.5>
    >>> loss_total(10, 8, 0.75)
    <tf.Tensor: shape=(), dtype=float32, numpy=9.5>
    """
    m_mae = tf.cast(m_mae, dtype=tf.float32)
    mae = tf.cast(mae, dtype=tf.float32)
    alpha_tensor = tf.cast(alpha, dtype=tf.float32)
    linear_complement = tf.math.subtract(tf.cast(1, dtype=tf.float32), alpha_tensor)
    return tf.add(tf.multiply(alpha_tensor, m_mae), tf.multiply(linear_complement, mae))


@tf.function
def loss_covariance_matrix(y_true, y_pred):
    """

    :param y_true: (BS, Frames, Rows, Columns, Channels+Masks)
    :param y_pred: (BS, Frames, Rows, Columns, Channels)
    :return:

    >>> pred = np.ones((16, 5, 32, 64, 2))
    >>> gt = np.ones((16, 5, 32, 64, 4))
    >>> mae = loss_covariance_matrix(gt, pred)
    >>> mae
    <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
    """

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_true = y_true[..., :2]  # Only use variables, no mask.
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    cm_samples_pred = reshape_batch_for_cm(y_pred)
    cm_samples_true = reshape_batch_for_cm(y_true)

    # Get the covariance matrix of the predictions
    pred_cm = calculate_covariance_matrix(cm_samples_pred)
    true_cm = calculate_covariance_matrix(cm_samples_true)

    # Calculate the MAE
    cm_mae = tf.reduce_mean(tf.abs(tf.subtract(true_cm, pred_cm)))

    return cm_mae


@tf.function
def loss_total2(total_loss, cm_loss, beta=0.9):
    """
    :param total_loss:
    :param cm_loss:
    :param beta:
    :return:

    >>> loss_total2(1, 1, 0.5)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
    >>> loss_total2(1, 1, 0.25)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
    >>> loss_total2(10, 8, 0.5)
    <tf.Tensor: shape=(), dtype=float32, numpy=9.0>
    >>> loss_total2(10, 8, 0.25)
    <tf.Tensor: shape=(), dtype=float32, numpy=8.5>
    >>> loss_total2(10, 8, 0.75)
    <tf.Tensor: shape=(), dtype=float32, numpy=9.5>
    """
    total_loss = tf.cast(total_loss, dtype=tf.float32)
    cm_loss = tf.cast(cm_loss, dtype=tf.float32)

    beta_tensor = tf.cast(beta, dtype=tf.float32)
    linear_complement = tf.math.subtract(tf.cast(1, dtype=tf.float32), beta_tensor)
    return tf.add(tf.multiply(beta_tensor, total_loss), tf.multiply(linear_complement, cm_loss))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
