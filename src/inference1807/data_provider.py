import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath("__init__.py"))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()
sys.path.append(os.path.dirname(SRC_DIR))

from src.config import METADATA_DIR
from src.config import VARIABLES_SET_INFERENCE, MASK_SET_INFERENCE, VARIABLES_SET_INFERENCE_ARM, \
                       MASK_SET_INFERENCE_ARM, VARIABLES_SET_INFERENCE_ARM_WT, MASK_SET_INFERENCE_ARM_WT
from src.config import F, H, W
from src.data_loaders.loading import get_elevation_data



def get_station_indices_map():
    """
    Returns dataframe referencing the station locations.
    :return:
    """
    df = pd.read_csv(str(SRC_DIR) + METADATA_DIR + "station_1807_matrix_indices.csv")
    return df


def _shift_samples(mat):
    """
    Shifts / reshapes samples by 1.
    :param mat: Samples of one year (1807) to be shifted by 1.
    :return:
    """
    shifted_samples = np.zeros((mat.shape[0] - F + 1, F, H, W, mat.shape[-1]))
    for i in range(shifted_samples.shape[0]):
        shifted_samples[i] = mat[i:i+F]
    return shifted_samples


def get_inference_x_data(missing_like):
    """
    Returns shifted x inference data depending on <missing_like>.
    :param missing_like: Str describing missing pattern. E.g. "plain"
    :return:
    """

    if missing_like == "plain":
        variables = np.load(str(SRC_DIR) + VARIABLES_SET_INFERENCE)
        masks = np.load(str(SRC_DIR) + MASK_SET_INFERENCE)
    elif missing_like == "arm":
        variables = np.load(str(SRC_DIR) + VARIABLES_SET_INFERENCE_ARM)
        masks = np.load(str(SRC_DIR) + MASK_SET_INFERENCE_ARM)
    elif missing_like == "arm_wt":
        variables = np.load(str(SRC_DIR) + VARIABLES_SET_INFERENCE_ARM_WT)
        masks = np.load(str(SRC_DIR) + MASK_SET_INFERENCE_ARM_WT)
    else:
        raise NotImplementedError(f"Your missing_like={missing_like} is not supported")

    elev = get_elevation_data(base_window_only=True)
    elev = np.expand_dims(elev, axis=(0, -1))
    elev = np.repeat(elev, repeats=variables.shape[0], axis=0)

    inf_data = np.concatenate([variables, masks, elev], axis=-1)
    inf_data = _shift_samples(inf_data)

    return inf_data
