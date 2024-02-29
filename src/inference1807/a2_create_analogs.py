"""
Arrangement Script 2
====================
Creates:
* List of analog days selected (metadata)
* Analog masks
* Analog enhanced variables
* Analog enhanced Weather Type Restricted variables
"""

import xarray as xr
import numpy as np
import pandas as pd
from itertools import product
import random

import os, sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath("__init__.py"))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()
sys.path.append(os.path.dirname(SRC_DIR))

from data_provider import get_station_indices_map
from data_transformer import extract_stations_from_nc

from src.config import METADATA_DIR
from src.config import TA_MU, TA_SIGMA, SLP_MU, SLP_SIGMA
from src.config import WINDOW_SIDE_SIZE
from src.config import H, W

from src.arm.api import (get_analog_pool_stations, get_analog_pool_windowed_ids,
                         get_observation_candidates_pairs, get_analog_ids, get_analog_pool,
                         get_analog_ids_with_same_weather_types)

TARGET_DATASETS_DIR = "data_sets/"
TARGET_MASK_DIR = "masks"


def _get_normalized_observation_stations(station_indx_map):
    observations = xr.load_dataset(F"{TARGET_DATASETS_DIR}/ground_truth.nc")
    obs_stations = extract_stations_from_nc(observations, station_indx_map)
    normalized_obs_stations = {}
    for k, v in obs_stations.items():
        if k.endswith("_ta"):
            normalized_obs_stations[k] = (v - TA_MU) / TA_SIGMA
        else:
            normalized_obs_stations[k] = (v - SLP_MU) / SLP_SIGMA
    return normalized_obs_stations


def main():
    percentage_to_sample = 3
    station_indx_map = get_station_indices_map()

    print("Read observations and analog pool")

    obs_stations = _get_normalized_observation_stations(station_indx_map)
    obs_dates = pd.date_range(start="1807-01-01", periods=365, freq="D")

    ap_stations = get_analog_pool_stations(station_indx_map)
    ap_dates = pd.date_range(start="1965-01-01", periods=len(list(ap_stations.values())[0]), freq="D")

    print("Get IDs of analog candidates")
    apw_ids = get_analog_pool_windowed_ids(obs_dates, ap_dates, WINDOW_SIDE_SIZE)

    obs_df = pd.DataFrame(data=obs_stations)
    ap_df = pd.DataFrame(data=ap_stations)

    print("Create observation candidate day pairs")
    observations_candidates = get_observation_candidates_pairs(obs_df, ap_df, apw_ids)

    # Create ARM and ARM-WT data in parallel! UGLY but keep it simple...

    print("Get nearest analogue neighbor WITH restriction on weather types")
    inf_weather_types = pd.read_csv(str(SRC_DIR) + METADATA_DIR + "inference1807_weather_types.csv")
    analog_weather_types = pd.read_csv(str(SRC_DIR) + METADATA_DIR + "analog_pool_weather_types.csv")
    analog_ids_wt, dists_wt = get_analog_ids_with_same_weather_types(observations_candidates, apw_ids,
                                                                     obs_wts=inf_weather_types,
                                                                     analog_wts=analog_weather_types)

    print("Get nearest analogue neighbor without restriction on weather types")
    analog_ids, dists = get_analog_ids(observations_candidates, apw_ids)

    print("Save analog dates")
    analog_df = pd.DataFrame({'analog_ids': analog_ids, 'analog_dates': [ap_dates[i] for i in analog_ids],
                              'dists': dists})
    analog_df_wt = pd.DataFrame({'analog_ids': analog_ids_wt, 'analog_dates': [ap_dates[i] for i in analog_ids_wt],
                                 'dists': dists_wt})

    analog_df.to_csv(f"arm_data/analog_days.csv", index_label="id")
    analog_df_wt.to_csv(f"arm_data/analog_days_WT.csv", index_label="id")

    print("Creating analog data sets")

    # Get selected analog days from the pool
    analog_pool = get_analog_pool()
    analog_ids = analog_df['analog_ids'].values
    analog_ids_wt = analog_df_wt['analog_ids'].values

    analog_day_set = analog_pool[analog_ids]
    analog_day_set_wt = analog_pool[analog_ids_wt]

    # Get locations (cells) that are not yet observed, to sample from.
    lats = list(range(H))
    lons = list(range(W))
    all_y_x_pairs = list(product(lats, lons))

    ta_obs_locs = list(
        station_indx_map[station_indx_map.variable == 'ta'][['y', 'x']].itertuples(index=False, name=None))
    slp_obs_locs = list(
        station_indx_map[station_indx_map.variable == 'slp'][['y', 'x']].itertuples(index=False, name=None))

    ta_locs_sample_space = list(set(all_y_x_pairs) - set(ta_obs_locs))
    slp_locs_sample_space = list(set(all_y_x_pairs) - set(slp_obs_locs))

    # Random sample locations and represent them in a mask
    addon_mask = np.ones_like(analog_day_set)  # SAME for ARM-WT!
    nbr_spatial_cells = addon_mask.shape[1] * addon_mask.shape[2]
    samples_to_draw = int((percentage_to_sample / 100) * nbr_spatial_cells)
    print("Number of samples to draw: ", samples_to_draw)

    for i in range(addon_mask.shape[0]):
        ta_day_msk_ids = random.choices(ta_locs_sample_space, k=samples_to_draw)
        slp_day_msk_ids = random.choices(slp_locs_sample_space, k=samples_to_draw)

        ta_ys = [loc[0] for loc in ta_day_msk_ids]
        ta_xs = [loc[1] for loc in ta_day_msk_ids]
        slp_ys = [loc[0] for loc in slp_day_msk_ids]
        slp_xs = [loc[1] for loc in slp_day_msk_ids]

        addon_mask[i, ta_ys, ta_xs, 0] = 0
        addon_mask[i, slp_ys, slp_xs, 1] = 0

    masked_analogs = np.ma.masked_array(analog_day_set, mask=addon_mask)
    masked_analogs_wt = np.ma.masked_array(analog_day_set_wt, mask=addon_mask)
    print(f"Addon mask contains {100 * (1 - addon_mask.mean()):.2f}% added observations.")

    # Load inference data
    mask_inf = np.load(f"{TARGET_MASK_DIR}/mask_set.npy")
    mask_inf_analog = np.copy(mask_inf)
    mask_inf_analog_wt = np.copy(mask_inf)

    variables_inf = np.load(f"{TARGET_DATASETS_DIR}/variables_set.npy")
    variables_inf_analog = np.copy(variables_inf)
    variables_inf_analog_wt = np.copy(variables_inf)

    # Apply addon mask and analog variables to inference data
    mask_inf_analog[~masked_analogs.mask] = masked_analogs.mask[~masked_analogs.mask]
    mask_inf_analog_wt[~masked_analogs_wt.mask] = masked_analogs_wt.mask[~masked_analogs_wt.mask]
    print(f"Resulting missing rate: {np.mean(mask_inf_analog)}")

    variables_inf_analog[~masked_analogs.mask] = masked_analogs.data[~masked_analogs.mask]
    variables_inf_analog_wt[~masked_analogs_wt.mask] = masked_analogs_wt.data[~masked_analogs_wt.mask]
    print(f"Resulting missing rate: {np.mean(mask_inf_analog)}")

    np.save(f"{TARGET_MASK_DIR}/mask_set_analog_{percentage_to_sample}p.npy",
            mask_inf_analog)
    np.save(f"{TARGET_DATASETS_DIR}/variables_set_analog_{percentage_to_sample}p.npy",
            variables_inf_analog)
    np.save(f"{TARGET_MASK_DIR}/mask_set_analog_{percentage_to_sample}p_WT.npy",
            mask_inf_analog_wt)
    np.save(f"{TARGET_DATASETS_DIR}/variables_set_analog_{percentage_to_sample}p_WT.npy",
            variables_inf_analog_wt)


if __name__ == "__main__":
    main()
