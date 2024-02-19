import mlflow

import tensorflow as tf
import numpy as np
import random


RANDOM_SEED = 42


def get_experiment(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        print(f"Experiment ({name}) is new, creating one...")
        exp_id = mlflow.create_experiment(name)
        return exp_id
    print(f"Experiment ({name}) already exists...")
    return exp.experiment_id


def set_random_state():
    """
    Sets random state for the relevant libraries to make experiments reproducible.
    :return:
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
