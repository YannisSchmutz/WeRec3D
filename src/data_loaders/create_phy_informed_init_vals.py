import os
from pathlib import Path
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()

DATA_DIR_A = str(SRC_DIR) + "/data_sets/variant_a/"
DATA_DIR_B = str(SRC_DIR) + "/data_sets/variant_b/"

TRAIN_SET_NAME = "train.npy"


def create_pi_init_vals(variant):
    print(f"[*] Create PI init vals for data variant {variant}")

    if variant == "a":
        base_dir = DATA_DIR_A
    elif variant == "b":
        base_dir = DATA_DIR_B
    else:
        raise RuntimeError(f"Variant {variant} not supported")

    train_variables = np.load(base_dir + TRAIN_SET_NAME)
    train_avg = np.mean(train_variables, axis=0)

    np.save(f"pi_init/averages_{variant}.npy", train_avg)


if __name__ == "__main__":
    create_pi_init_vals("a")
    create_pi_init_vals("b")
