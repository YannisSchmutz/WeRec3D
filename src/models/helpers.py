import tensorflow as tf
import numpy as np


def calculate_covariance_matrix(tensor):
    """
    Estimate a covariance matrix. Each column represents a variable (pixels), with observations (days) in the rows.
    Thus, the input tensor is in the form $I^{(days, flatted-fields)}$

    :param tensor:
    :return:

    >>> #
    >>> flat_samples = np.vstack((np.arange(0, 5), np.arange(0, 5)+1, np.arange(0, 5)+2))
    >>> cm = calculate_covariance_matrix(flat_samples)
    >>> cm.shape
    TensorShape([5, 5])
    >>> cm
    <tf.Tensor: shape=(5, 5), dtype=float32, numpy=
    array([[1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.]], dtype=float32)>
    >>> # Second test scenario
    >>> day1 = np.array([0,0,0, 1,1,1, 2,2,2])
    >>> day2 = np.array([1,0,1, 1,1,1, 1,2,1])
    >>> day3 = np.array([2,0,2, 1,1,1, 0,2,0])
    >>> day4 = np.array([3,0,3, 1,1,1, -1,2,-1])
    >>> seq = np.vstack((day1, day2, day3, day4))
    >>> cm = calculate_covariance_matrix(seq)
    >>> cm.shape
    TensorShape([9, 9])
    >>> cm = cm.numpy()
    >>> print([round(cm[0, j], 1) for j in range(9)])
    [1.7, 0.0, 1.7, 0.0, 0.0, 0.0, -1.7, 0.0, -1.7]
    >>> print([round(cm[1, j], 1) for j in range(9)])
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    >>> print([round(cm[2, j], 1) for j in range(9)])
    [1.7, 0.0, 1.7, 0.0, 0.0, 0.0, -1.7, 0.0, -1.7]
    """
    #
    tensor = tf.cast(tensor, dtype=tf.float32)

    # Calculate the mean for each variable, over the samples (rows -> axis=0)
    variable_means = tf.reduce_mean(tensor, axis=0)

    # Subtract the mean along each column
    mean_centered = tf.subtract(tensor, variable_means)

    # Calculate the covariance matrix
    _x = tf.transpose(mean_centered)
    _y = mean_centered
    _n = tf.cast(tf.shape(tensor)[0], dtype=tf.float32)
    cov_matrix = tf.matmul(_x, _y) / (_n - 1)

    return cov_matrix


@tf.function
def reshape_batch_for_cm(batch):
    """

    :param batch:
    :return:
    >>> b = np.ones((16, 5, 32, 64, 2))
    >>> s = reshape_batch_for_cm(b)
    >>> s.shape
    TensorShape([80, 4096])
    >>> b0 = np.zeros((16, 5, 32, 64, 1))
    >>> b1 = np.ones((16, 5, 32, 64, 1))
    >>> b = np.concatenate((b0, b1), axis=-1)
    >>> s = reshape_batch_for_cm(b)
    >>> (s.numpy()[:, :2048] == 0).all()
    True
    >>> (s.numpy()[:, 2048:] == 1).all()
    True
    """
    bs = tf.shape(batch)[0]
    f = tf.shape(batch)[1]
    h = tf.shape(batch)[2]
    w = tf.shape(batch)[3]
    ch = tf.shape(batch)[4]

    # Reshape to get all days in one dimension
    samples = tf.reshape(batch, (bs * f, h, w, ch))

    # For each day, create a row-vector of "pixels"
    flat_t2m = tf.reshape(samples[..., 0], (bs * f, h * w))
    flat_msl = tf.reshape(samples[..., 1], (bs * f, h * w))

    # Add them together horizontally, expanding the variables
    flat_samples = tf.stack([flat_t2m, flat_msl], axis=1)
    flat_samples = tf.reshape(flat_samples, (bs * f, h * w * ch))

    return flat_samples


if __name__ == '__main__':
    import doctest
    doctest.testmod()
