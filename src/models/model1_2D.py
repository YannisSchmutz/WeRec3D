import tensorflow as tf
from keras.models import Model
from keras.layers import Input

try:
    #from src.models.metrics import masked_mae, overall_mae
    # from src.models.losses import loss_total
    from src.models.layers_model_2D import create_model_layers
except ImportError:
    # from models.metrics import masked_mae, overall_mae
    # from models.losses import loss_total
    from models.layers_model_2D import create_model_layers


@tf.function
def masked_mae(y_true, y_pred):
    if y_true.dtype != y_pred.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)

    # BS, H, W, CH
    ch = tf.shape(y_true)[3]

    half_ch = tf.dtypes.cast(tf.divide(ch, 2), tf.int32)
    masks = y_true[..., half_ch:]
    y_true = y_true[..., :half_ch]

    numerators = tf.math.subtract(y_pred, y_true)
    numerators = tf.math.multiply(masks, numerators)
    numerators = tf.math.abs(numerators)
    # Sum has to be done per sample in batch, per frame in sample -> leave out first and second dim
    numerators = tf.math.reduce_sum(numerators, axis=tf.range(start=1, limit=len(tf.shape(numerators))))

    denominators = tf.math.reduce_sum(masks, axis=tf.range(start=1, limit=len(tf.shape(masks))))

    # Would fail if mask was 0%
    quotients = tf.math.divide(numerators, denominators)

    averaged_frame_sum = tf.math.reduce_mean(quotients, axis=-1)
    averaged_batch_sum = tf.math.reduce_mean(averaged_frame_sum)
    return averaged_batch_sum


@tf.function
def overall_mae(y_true, y_pred):
    if y_true.dtype != y_pred.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)

    ch = tf.shape(y_true)[3]

    half_ch = tf.dtypes.cast(tf.divide(ch, 2), tf.int32)
    y_true = y_true[..., :half_ch]

    mae = tf.math.reduce_mean(
        tf.math.abs(
            tf.math.subtract(y_pred, y_true)
        )
    )
    return mae


@tf.function
def loss_total(m_mae, mae, alpha=0.5):
    m_mae = tf.cast(m_mae, dtype=tf.float32)
    mae = tf.cast(mae, dtype=tf.float32)
    alpha_tensor = tf.cast(alpha, dtype=tf.float32)
    linear_complement = tf.math.subtract(tf.cast(1, dtype=tf.float32), alpha_tensor)
    return tf.add(tf.multiply(alpha_tensor, m_mae), tf.multiply(linear_complement, mae))


class Model2D(Model):

    def __init__(self, *args, **kwargs):
        super(Model2D, self).__init__(*args, **kwargs)

        # The trackers are used to get the mean value after each epoch
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.mae_tracker = tf.keras.metrics.Mean(name='mae')
        self.masked_mae_tracker = tf.keras.metrics.Mean(name='masked_mae')

    def _get_config(self):
        # mlflow tries to access this private method...
        return self.get_config()

    def get_config(self):
        """
        Saves constructor args. Required for saving the model...
        :return:
        """
        config = super().get_config()
        # save constructor args
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    @property
    def metrics(self):
        """
        Overwrites parent method.

        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        :return:
        """
        return [self.loss_tracker, self.masked_mae_tracker, self.mae_tracker]

    def train_step(self, data):
        xb, yb = data

        with tf.GradientTape() as tape:
            pred = self(xb, training=True)
            m_mae = masked_mae(yb, pred)
            mae = overall_mae(yb, pred)
            total_loss = loss_total(m_mae, mae)

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.loss_tracker.update_state(total_loss)
        self.masked_mae_tracker.update_state(m_mae)
        self.mae_tracker.update_state(mae)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        xb, yb = data
        pred = self(xb, training=False)
        m_mae = masked_mae(yb, pred)
        mae = overall_mae(yb, pred)
        total_loss = loss_total(m_mae, mae)

        # Automatically creates "val_loss", "val_masked_mae"
        self.loss_tracker.update_state(total_loss)
        self.masked_mae_tracker.update_state(m_mae)
        self.mae_tracker.update_state(mae)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


def create_model(*, h, w, ch, bs):

    # Fix batch_size since cn2d needs to know it in advance...
    input_sequence = Input(shape=(h, w, ch), batch_size=bs)

    model_layer_tensor = create_model_layers(input_sequence)
    model = Model2D(inputs=input_sequence, outputs=model_layer_tensor, name='Model2D')
    # model.summary()

    return model
