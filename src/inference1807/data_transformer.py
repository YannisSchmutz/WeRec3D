import os, sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath("__init__.py"))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()
sys.path.append(os.path.dirname(SRC_DIR))

from src.config import TA_MU, TA_SIGMA, SLP_MU, SLP_SIGMA
from src.config import F, H, W  # DAYS_IN_1807
from src.config import BASE_LAT_START, BASE_LAT_END, BASE_LON_START, BASE_LON_END


def standardize_data(data, mu, sigma):
    """
    Performs z-normalization.
    :param data: Data to be scaled
    :param mu: Mean
    :param sigma: Standard deviation
    :return:
    """
    data = (data - mu) / sigma
    return data


def scale_back(var_mat):
    """
    Scales variables back to °C and Pa.
    :param var_mat: Numpy matrix with shape (Days, Height, Width, Channels)
    :return:
    """
    ta_pred = var_mat[..., 0]
    slp_pred = var_mat[..., 1]
    ta_pred = np.add(np.multiply(ta_pred, TA_SIGMA), TA_MU)
    slp_pred = np.add(np.multiply(slp_pred, SLP_SIGMA), SLP_MU)
    back_scaled_pred = np.concatenate([np.expand_dims(ta_pred, axis=-1),
                                       np.expand_dims(slp_pred, axis=-1)
                                       ], axis=-1)
    return back_scaled_pred


def extract_stations_from_nc(data_set, stat_indx_map):
    """
    Returns dict of only station time series of a corresponding xarray data set.
    :param data_set: xarray dataset (could be prediction or gt)
    :param stat_indx_map: Dataframe referencing the station locations.
    :return:
    """
    stations = {}
    for index in stat_indx_map.index:
        station_id = stat_indx_map.loc[index].id
        variable = stat_indx_map.loc[index].variable
        lat = stat_indx_map.loc[index].lat
        lon = stat_indx_map.loc[index].lon

        # just time dim.
        station_val = data_set[variable].sel(dict(lat=lat, lon=lon)).to_numpy()
        stations[f"{station_id}_{variable}"] = station_val
    return stations


def get_median_pred_days(pred):
    """
    Apply rolling median over prediction to get the median values for each day.
    :param pred: Prediction with shape(Samples, Frames, H, W, C)
    :return:
    """
    days_in_1807 = 365
    median_pred = np.zeros((days_in_1807, H, W, 2))
    for i in range(days_in_1807):
        sample_id = i
        frame_offset = 0
        if i > days_in_1807 - F:
            sample_id = days_in_1807 - F
            frame_offset = i - (days_in_1807 - F)

        window_sample_ids = np.arange(sample_id, sample_id-F, -1)
        window_sample_ids = window_sample_ids[window_sample_ids >= 0]
        corresp_frame_ids = np.array([j+frame_offset for j in range(len(window_sample_ids))])
        corresp_frame_ids = corresp_frame_ids[corresp_frame_ids < F]
        window_sample_ids = window_sample_ids[:len(corresp_frame_ids)]

        day_i_val = np.median(pred[window_sample_ids, corresp_frame_ids], axis=0)
        median_pred[i] = day_i_val
    return median_pred

