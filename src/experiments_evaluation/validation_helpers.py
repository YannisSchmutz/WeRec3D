import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath("__init__.py"))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()
sys.path.append(os.path.dirname(SRC_DIR))

from src.config import TA_MU, TA_SIGMA, SLP_MU, SLP_SIGMA
from src.config import F, H, W
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


def reshape_for_modelling(mat, seq_shift_reshape=True):
    base_shape = mat.shape

    # Crop to base window
    mat = mat[:, BASE_WINDOW_Y_CROP_1:BASE_WINDOW_Y_CROP_2, BASE_WINDOW_X_CROP_1:BASE_WINDOW_X_CROP_2]

    nbr_samples = base_shape[0] // F
    cut_off = nbr_samples * F
    # Make sure it is dividable through "f"
    mat = mat[:cut_off]
    if seq_shift_reshape:
        shifted_samples = np.zeros((mat.shape[0] - F + 1, F, H, W, mat.shape[-1]))
        for i in range(shifted_samples.shape[0]):
            shifted_samples[i] = mat[i:i + F]
        return shifted_samples
    return mat


def get_median_pred_days(pred):
    """
    Apply rolling median over prediction to get the median values for each day.
    :param pred: Prediction with shape(Samples, Frames, H, W, C)
    :return:
    """
    days_in_pred = pred.shape[0]
    median_pred = np.zeros((days_in_pred, H, W, 2))
    for i in range(days_in_pred):
        sample_id = i
        frame_offset = 0
        if i > days_in_pred - F:
            sample_id = days_in_pred - F
            frame_offset = i - (days_in_pred - F)

        window_sample_ids = np.arange(sample_id, sample_id-F, -1)
        window_sample_ids = window_sample_ids[window_sample_ids >= 0]
        corresp_frame_ids = np.array([j+frame_offset for j in range(len(window_sample_ids))])
        corresp_frame_ids = corresp_frame_ids[corresp_frame_ids < F]
        window_sample_ids = window_sample_ids[:len(corresp_frame_ids)]

        day_i_val = np.median(pred[window_sample_ids, corresp_frame_ids], axis=0)
        median_pred[i] = day_i_val
    return median_pred


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
