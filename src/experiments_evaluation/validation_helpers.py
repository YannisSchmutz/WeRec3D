import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath("__init__.py"))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()
sys.path.append(os.path.dirname(SRC_DIR))

from src.config import TA_MU, TA_SIGMA, SLP_MU, SLP_SIGMA
from src.config import BASE_WINDOW_Y_CROP_1, BASE_WINDOW_Y_CROP_2, BASE_WINDOW_X_CROP_1, BASE_WINDOW_X_CROP_2

import numpy as np


def scale_t2m_back(var, for_error=False):
    if for_error:
        # Scaling back for error only requires sigma, the mu gets eliminated in the subtraction
        return np.multiply(var, TA_SIGMA)
    return np.add(np.multiply(var, TA_SIGMA),  TA_MU)


def scale_slp_back(var, for_error=False):
    if for_error:
        # Scaling back for error only requires sigma, the mu gets eliminated in the subtraction
        return np.multiply(var, SLP_SIGMA)
    return np.add(np.multiply(var, SLP_SIGMA),  SLP_MU)


def prepare_quantitative_samples1(mat, f=5, seq_reshape=True):
    """

    Expected shape: (Time, Hight, Width, Channels)
    """
    # Crop to base window
    mat = mat[:, BASE_WINDOW_Y_CROP_1:BASE_WINDOW_Y_CROP_2, BASE_WINDOW_X_CROP_1:BASE_WINDOW_X_CROP_2]
    base_shape = mat.shape

    nbr_samples = base_shape[0] // f
    cut_off = nbr_samples * f
    # Make sure it is dividable through "f"
    mat = mat[:cut_off]
    # Reshape to samples, frames, ...
    if seq_reshape:
        mat = np.reshape(mat, (nbr_samples, f, base_shape[1], base_shape[2], base_shape[3]))
    return mat


def calc_total_errors(y, pred):
    abs_err = np.abs(np.subtract(pred, y))

    total_err = np.mean(abs_err)
    total_variable_error = np.mean(abs_err, axis=tuple([shp for shp in range(len(abs_err.shape) - 1)]))
    return total_err, total_variable_error[0], total_variable_error[1]


def calc_spatial_errors(y, pred):
    abs_err = np.abs(np.subtract(pred, y))
    spatial_err = np.mean(abs_err, axis=(0, -1))
    spatial_variable_error = np.mean(abs_err, axis=0)
    return spatial_err, spatial_variable_error[..., 0], spatial_variable_error[..., 1]


def calc_temporal_errors(y, pred):
    abs_err = np.abs(np.subtract(pred, y))
    temporal_error = np.mean(abs_err, axis=tuple([shp for shp in range(1, len(abs_err.shape))]))
    temporal_variable_error = np.mean(abs_err, axis=tuple([shp for shp in range(1, len(abs_err.shape) - 1)]))
    return temporal_error, temporal_variable_error[..., 0], temporal_variable_error[..., 1]
