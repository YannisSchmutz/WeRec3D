import tensorflow as tf
from keras.models import Model
from keras.layers import Input

try:
    from src.models.metrics import masked_mae, overall_mae, masked_rmse, overall_rmse
    from src.models.losses import loss_total, loss_total2, loss_covariance_matrix
    from src.models.layers_model_1 import create_model_layers
except ImportError:
    from models.metrics import masked_mae, overall_mae, masked_rmse, overall_rmse
    from models.losses import loss_total, loss_total2, loss_covariance_matrix
    from models.layers_model_1 import create_model_layers


class Model2(Model):

    def __init__(self, cm_beta, *args, alpha=0.5, **kwargs):
        super(Model2, self).__init__(*args, **kwargs)

        self.cm_beta = cm_beta
        self.alpha = alpha

        # The trackers are used to get the mean value after each epoch
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.basic_loss_tracker = tf.keras.metrics.Mean(name='basic_loss')
        self.mae_tracker = tf.keras.metrics.Mean(name='mae')
        self.masked_mae_tracker = tf.keras.metrics.Mean(name='masked_mae')
        self.rmse_tracker = tf.keras.metrics.Mean(name='rmse')
        self.masked_rmse_tracker = tf.keras.metrics.Mean(name='masked_rmse')
        self.cm_loss_tracker = tf.keras.metrics.Mean(name='cm_loss')

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
        return [self.loss_tracker, self.basic_loss_tracker,
                self.masked_mae_tracker, self.mae_tracker, self.cm_loss_tracker,
                self.rmse_tracker, self.masked_rmse_tracker]

    def train_step(self, data):
        xb, yb = data

        with tf.GradientTape() as tape:
            pred = self(xb, training=True)
            m_mae = masked_mae(yb, pred)
            mae = overall_mae(yb, pred)
            basic_loss = loss_total(m_mae, mae, alpha=self.alpha)
            cm_loss = loss_covariance_matrix(yb, pred)
            total_loss = loss_total2(basic_loss, cm_loss, self.cm_beta)

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.loss_tracker.update_state(total_loss)
        self.basic_loss_tracker.update_state(basic_loss)
        self.masked_mae_tracker.update_state(m_mae)
        self.mae_tracker.update_state(mae)
        self.cm_loss_tracker.update_state(cm_loss)

        rmse = overall_rmse(yb, pred)
        m_rmse = masked_rmse(yb, pred)
        self.rmse_tracker.update_state(rmse)
        self.masked_rmse_tracker.update_state(m_rmse)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        xb, yb = data
        pred = self(xb, training=False)
        m_mae = masked_mae(yb, pred)
        mae = overall_mae(yb, pred)
        basic_loss = loss_total(m_mae, mae, alpha=self.alpha)
        cm_loss = loss_covariance_matrix(yb, pred)
        total_loss = loss_total2(basic_loss, cm_loss, self.cm_beta)
        rmse = overall_rmse(yb, pred)
        m_rmse = masked_rmse(yb, pred)

        # Automatically creates "val_loss", "val_masked_mae"
        self.loss_tracker.update_state(total_loss)
        self.basic_loss_tracker.update_state(basic_loss)
        self.masked_mae_tracker.update_state(m_mae)
        self.mae_tracker.update_state(mae)
        self.cm_loss_tracker.update_state(cm_loss)
        self.rmse_tracker.update_state(rmse)
        self.masked_rmse_tracker.update_state(m_rmse)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


def create_model(*, f, h, w, ch, bs, alpha=0.5, cm_beta=0.9, af_variant=1):

    # Fix batch_size since cn3d needs to know it in advance...
    input_sequence = Input(shape=(f, h, w, ch), batch_size=bs)

    model_layer_tensor = create_model_layers(input_sequence, af_variant=af_variant)
    model = Model2(cm_beta, alpha=alpha, inputs=input_sequence, outputs=model_layer_tensor, name='Model2')
    # model.summary()

    return model
