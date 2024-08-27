import numpy as np
import tensorflow as tf


@tf.function
def masked_mae(y_true, y_pred):
    """
    Custom MAE metric.

    Mathematical definition:
    $$ \text{masked_mae}(Y, \hat{Y}, M) = \frac{1}{BS}\frac{1}{F} \sum_{b=1}^{BS}\sum_{k=1}^{F} \frac{|| M_{b}^{k} \odot (Y_{b}^{k} - \hat{Y}_{b}^{k}) ||}{|| M_{b}^{k} ||} $$

    :param y_true: Batch of ground truth with mask as last channel.
                   Shape: (BS, Frames, Rows, Columns, Channels+Masks)
    :param y_pred: Batch of predicted rectangle.
                   Shape: (BS, Frames, Rows, Columns, Channels)
    :return:

    # === One-Channel Variables Tests ====
    # NOT supported ATM

    >>> # === Three-Channel Variable Tests ====
    >>> # Expect zero
    >>> pred = np.ones((4, 32, 64, 64, 3))
    >>> gt = np.ones((4, 32, 64, 64, 6)) # Include masks right here
    >>> masked_mae(gt, pred)
    <tf.Tensor: shape=(), dtype=float64, numpy=0.0>
    >>> # Expected loss is three times higher than for one-channeled data
    >>> m_sample1 = np.array([[[1,0], [1,0]], [[1,1], [0,1]]])
    >>> m_sample1 = np.expand_dims(m_sample1, axis=-1)
    >>> m_sample1 = np.repeat(m_sample1, 3, axis=-1)
    >>> m_sample2 = np.array([[[1,0], [1,0]], [[1,1], [0,1]]])
    >>> m_sample2 = np.expand_dims(m_sample2, axis=-1)
    >>> m_sample2 = np.repeat(m_sample2, 3, axis=-1)
    >>> ms = np.stack((m_sample1, m_sample2), axis=0)
    >>> gt_sample1 = np.repeat(np.expand_dims(np.array([[[1,2],[3,4]], [[5,6],[7,8]]]), axis=-1), 3, axis=-1)
    >>> gt_sample2 = np.repeat(np.expand_dims(np.array([[[10,20],[30,40]], [[50,60],[70,80]]]), axis=-1), 3, axis=-1)
    >>> gt = np.stack((gt_sample1, gt_sample2), axis=0)
    >>> gt = np.concatenate((gt, ms), axis=-1)
    >>> pred1 = np.repeat(np.expand_dims(np.array([[[11,2],[3,4]], [[15,6],[7,8]]]), axis=-1), 3, axis=-1)
    >>> pred2 = np.repeat(np.expand_dims(np.array([[[10,20],[40,40]], [[50,50],[70,80]]]), axis=-1), 3, axis=-1)
    >>> pred = np.stack((pred1, pred2), axis=0)
    >>> masked_mae(gt, pred)
    <tf.Tensor: shape=(), dtype=float64, numpy=4.166666666666667>
    >>> # === Four-Channel Variable Tests ====
    >>> # Expect zero
    >>> pred = np.ones((4, 32, 64, 64, 4))
    >>> gt = np.ones((4, 32, 64, 64, 8)) # Include mask right here
    >>> masked_mae(gt, pred)
    <tf.Tensor: shape=(), dtype=float64, numpy=0.0>
    >>> # Expected loss is three times higher than for one-channeled data
    >>> m_sample1 = np.array([[[1,0], [1,0]], [[1,1], [0,1]]])
    >>> m_sample1 = np.expand_dims(m_sample1, axis=-1)
    >>> m_sample1 = np.repeat(m_sample1, 4, axis=-1)
    >>> m_sample2 = np.array([[[1,0], [1,0]], [[1,1], [0,1]]])
    >>> m_sample2 = np.expand_dims(m_sample2, axis=-1)
    >>> m_sample2 = np.repeat(m_sample2, 4, axis=-1)
    >>> ms = np.stack((m_sample1, m_sample2), axis=0)
    >>> gt_sample1 = np.repeat(np.expand_dims(np.array([[[1,2],[3,4]], [[5,6],[7,8]]]), axis=-1), 4, axis=-1)
    >>> gt_sample2 = np.repeat(np.expand_dims(np.array([[[10,20],[30,40]], [[50,60],[70,80]]]), axis=-1), 4, axis=-1)
    >>> gt = np.stack((gt_sample1, gt_sample2), axis=0)
    >>> gt = np.concatenate((gt, ms), axis=-1)
    >>> pred1 = np.repeat(np.expand_dims(np.array([[[11,2],[3,4]], [[15,6],[7,8]]]), axis=-1), 4, axis=-1)
    >>> pred2 = np.repeat(np.expand_dims(np.array([[[10,20],[40,40]], [[50,50],[70,80]]]), axis=-1), 4, axis=-1)
    >>> pred = np.stack((pred1, pred2), axis=0)
    >>> masked_mae(gt, pred)
    <tf.Tensor: shape=(), dtype=float64, numpy=4.166666666666667>
    """
    if y_true.dtype != y_pred.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)

    ch = tf.shape(y_true)[4]

    half_ch = tf.dtypes.cast(tf.divide(ch, 2), tf.int32)
    masks = y_true[..., half_ch:]
    y_true = y_true[..., :half_ch]

    numerators = tf.math.subtract(y_pred, y_true)
    numerators = tf.math.multiply(masks, numerators)
    numerators = tf.math.abs(numerators)
    # Sum has to be done per sample in batch, per frame in sample -> leave out first and second dim
    numerators = tf.math.reduce_sum(numerators, axis=tf.range(start=2, limit=len(tf.shape(numerators))))

    denominators = tf.math.reduce_sum(masks, axis=tf.range(start=2, limit=len(tf.shape(masks))))

    # Would fail if mask was 0%
    quotients = tf.math.divide(numerators, denominators)

    averaged_frame_sum = tf.math.reduce_mean(quotients, axis=-1)
    averaged_batch_sum = tf.math.reduce_mean(averaged_frame_sum)
    return averaged_batch_sum


@tf.function
def overall_mae(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:

    >>> # === Three-Channel Variable Tests ====
    >>> # Expect zero
    >>> pred = np.ones((4, 32, 64, 64, 3))
    >>> gt = np.ones((4, 32, 64, 64, 6)) # Include masks right here
    >>> overall_mae(gt, pred)
    <tf.Tensor: shape=(), dtype=float64, numpy=0.0>
    >>> pred = np.zeros((4, 32, 64, 64, 3))
    >>> gt = np.ones((4, 32, 64, 64, 6))
    >>> overall_mae(gt, pred)
    <tf.Tensor: shape=(), dtype=float64, numpy=1.0>
    >>> pred = np.zeros((4, 32, 64, 64, 3))
    >>> pred += 1.5
    >>> gt = np.ones((4, 32, 64, 64, 6))
    >>> overall_mae(gt, pred)
    <tf.Tensor: shape=(), dtype=float64, numpy=0.5>
    """
    if y_true.dtype != y_pred.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)

    ch = tf.shape(y_true)[4]

    half_ch = tf.dtypes.cast(tf.divide(ch, 2), tf.int32)
    y_true = y_true[..., :half_ch]

    mae = tf.math.reduce_mean(
        tf.math.abs(
            tf.math.subtract(y_pred, y_true)
        )
    )
    return mae


@tf.function
def overall_rmse(y_true, y_pred):
    """
    >>> # === Three-Channel Variable Tests ====
    >>> # Expect zero
    >>> pred = np.ones((4, 32, 64, 64, 3))
    >>> gt = np.ones((4, 32, 64, 64, 6)) # Include masks right here
    >>> overall_mae(gt, pred)
    <tf.Tensor: shape=(), dtype=float64, numpy=0.0>
    >>> pred = np.zeros((4, 32, 64, 64, 3))
    >>> gt = np.ones((4, 32, 64, 64, 6))
    >>> overall_mae(gt, pred)
    <tf.Tensor: shape=(), dtype=float64, numpy=1.0>
    >>> pred = np.zeros((4, 32, 64, 64, 3))
    >>> pred += 1.5
    >>> gt = np.ones((4, 32, 64, 64, 6))
    >>> overall_mae(gt, pred)
    <tf.Tensor: shape=(), dtype=float64, numpy=0.5>
    """
    if y_true.dtype != y_pred.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)
    ch = tf.shape(y_true)[4]

    half_ch = tf.dtypes.cast(tf.divide(ch, 2), tf.int32)
    y_true = y_true[..., :half_ch]

    # Calculate the squared differences
    squared_diffs = tf.square(y_true - y_pred)
    # Calculate the mean of the squared differences
    mean_squared_diff = tf.reduce_mean(squared_diffs)
    # Calculate the square root of the mean squared differences (RMSE)
    return tf.sqrt(mean_squared_diff)


@tf.function
def masked_rmse(y_true, y_pred):
    """
    >>> gt = np.array([[2.0,4.0], [8.0, 16.0]])  # h, w
    >>> gt = np.expand_dims(gt, axis=-1)
    >>> gt = np.repeat(gt, 2, axis=-1)
    >>> gt = np.expand_dims(gt, axis=0)
    >>> gt = np.repeat(gt, 2, axis=0)
    >>> gt = np.expand_dims(gt, axis=0)
    >>> gt = np.repeat(gt, 2, axis=0)
    >>> msk = np.ones_like(gt)
    >>> gt = np.concatenate((gt, msk), axis=-1)
    >>> pred = np.array([[1.0, 2.0], [5.0, 12.0]])
    >>> pred = np.expand_dims(pred, axis=-1)
    >>> pred = np.repeat(pred, 2, axis=-1)
    >>> pred = np.expand_dims(pred, axis=0)
    >>> pred = np.repeat(pred, 2, axis=0)
    >>> pred = np.expand_dims(pred, axis=0)
    >>> pred = np.repeat(pred, 2, axis=0)
    >>> masked_rmse(gt, pred)
    <tf.Tensor: shape=(), dtype=float64, numpy=2.7386127875258306>
    >>> gt[:, :, 1, ...] = 0
    >>> masked_rmse(gt, pred)
    <tf.Tensor: shape=(), dtype=float64, numpy=1.5811388300841898>
    """
    if y_true.dtype != y_pred.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)
    ch = tf.shape(y_true)[4]

    half_ch = tf.dtypes.cast(tf.divide(ch, 2), tf.int32)
    masks = y_true[..., half_ch:]
    y_true = y_true[..., :half_ch]

    # Calculate the squared differences
    squared_diffs = tf.square(y_true - y_pred)
    # Only keep values where mask == 1, the others are multiplied by zero.
    masked_squared_diffs = tf.math.multiply(masks, squared_diffs)
    mean_squared_diff = tf.reduce_sum(masked_squared_diffs) / tf.reduce_sum(masks)
    m_rmse = tf.sqrt(mean_squared_diff)

    return m_rmse


if __name__ == "__main__":
    import doctest
    doctest.testmod()
