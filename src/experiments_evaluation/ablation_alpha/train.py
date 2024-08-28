import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
SRC_DIR = Path(SCRIPT_DIR).parent.parent.absolute()
print(SRC_DIR)
sys.path.append(os.path.dirname(SRC_DIR))
sys.path.append(os.path.dirname(str(SRC_DIR) + '/models'))

import click
from contextlib import nullcontext
import mlflow
import tensorflow as tf
from pprint import pprint as pp

try:
    from src.models.model1 import create_model
    from src.data_loaders.loading import get_train_val_sets
    from src.data_loaders.generators import DataGenerator1
    from src.experiments_evaluation.experiment_management import get_experiment, set_random_state
    from src.experiments_evaluation.experiment_descriptions import EXPERIMENTS
except ImportError:
    from models.model1 import create_model
    from data_loaders.loading import get_train_val_sets
    from data_loaders.generators import DataGenerator1
    from experiments_evaluation.experiment_management import get_experiment, set_random_state
    from experiments_evaluation.experiment_descriptions import EXPERIMENTS

F = 5
H = 32
W = 64
CH = 4  # t2m, msl, msk1, msk2
BS = 32 # One worker
EPOCHS = 10

DATA_VARIANT = "b"  # With seasonality
CHECKPOINT_FILE_PATH = "model_checkpoint/"


@click.command()
@click.option('-n', '--exp_name', required=True, type=str)
@click.option('-a', '--alpha', required=True, type=float)
@click.option('-p', '--percentage', required=True, type=str)
@click.option('-l', '--load_last', required=True, type=str)
def main(exp_name, alpha, percentage, load_last):
    # SET MLFLOW dir
    mlflow.set_tracking_uri(f"file://{SRC_DIR}/experiments_evaluation/mlruns")

    experiment_id = get_experiment(exp_name)
    print(f"[*] Using experiment: {exp_name}")
    experiment_metadata = {"desc": "Investigate different alpha values",
                           "model": "model1",
                           "loss": "",
                           "hyperparameters": "default",
                           "seasonal": "",
                           }

    print("[*] Experiment metadata:")
    pp(experiment_metadata)

    mlflow.tensorflow.autolog(
        log_models=False,  # No need for storing the keras models. We handle this via checkpoint-callback.
        log_model_signatures=False  # Don't know exactly what this does, but prevents mlflow-warning...
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    # Manually set the name of the run
    mlflow.set_tag("mlflow.runName", f"inc_pretrain_{percentage}p")

    gpu_available = bool(tf.config.list_physical_devices('GPU'))

    set_random_state()

    train_val_sets = get_train_val_sets(variant=DATA_VARIANT, percentage=percentage)
    X_train = train_val_sets['train']['x']
    y_train = train_val_sets['train']['y']
    X_validation = train_val_sets['validation']['x']
    y_validation = train_val_sets['validation']['y']
    print(f"[*] Train X shape: {X_train.shape}")
    print(f"[*] Train y shape: {y_train.shape}")
    print(f"[*] Validation x shape: {X_validation.shape}")
    print(f"[*] Validation y shape: {y_validation.shape}")

    train_generator = DataGenerator1(X_train, y_train, batch_size=BS, seq_length=F)
    validation_generator = DataGenerator1(X_validation, y_validation, batch_size=BS, seq_length=F)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope() if gpu_available else nullcontext():
        model = create_model(f=F, h=H, w=W, ch=CH, bs=BS, alpha=alpha)
        # Default: beta_1=0.9, beta_2=0.999, learning_rate=0.001
        model.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly=None)
        if load_last:
            print(f"[*] Loading the weights of the last trained model ({load_last}%)")
            model.load_weights(f"{CHECKPOINT_FILE_PATH}p{load_last}_alpha{alpha}.weights.h5")

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{CHECKPOINT_FILE_PATH}p{percentage}_alpha{alpha}.weights.h5",
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    print("[*] Created model, start training")
    fit_ret = model.fit(train_generator,
                        validation_data=validation_generator,
                        epochs=EPOCHS,
                        callbacks=[model_checkpoint_callback])
    print("[*] Finished training")
    print(fit_ret.history)

    # RUN is already active when the name is set at the beginning
    mlflow.log_dict(experiment_metadata, 'meta_data.json')
    mlflow.log_param('percentage', percentage)
    mlflow.log_param('alpha', alpha)


if __name__ == '__main__':
    main()
