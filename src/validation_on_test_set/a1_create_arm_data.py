import numpy as np
from itertools import product
import pandas as pd
import random

import os, sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath("__init__.py"))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()
sys.path.append(os.path.dirname(SRC_DIR))

from src.config import H, W
from src.config import BASE_WINDOW_Y_CROP_1, BASE_WINDOW_Y_CROP_2, BASE_WINDOW_X_CROP_1, BASE_WINDOW_X_CROP_2
from src.config import ORIG_TEST_SET
from src.config import WINDOW_SIDE_SIZE
from src.config import T2M_REPLACE_VALUE_B, SLP_REPLACE_VALUE_B
from src.config import METADATA_DIR

from src.arm.api import (extract_stations_from_npy, get_analog_pool_stations, get_analog_pool_windowed_ids,
                         get_observation_candidates_pairs, get_analog_ids, get_analog_pool,
                         get_analog_ids_with_same_weather_types)


TARGET_DATASETS_DIR = "arm_data/"


# Find analogs for test data
def get_station_indices_map():
    """
    Returns dataframe referencing the station locations depending.
    :return:
    """
    df = pd.read_csv(str(SRC_DIR) + METADATA_DIR + "station_1807_matrix_indices.csv")
    return df


def main():
    """

    :param restrict_weather_types:
    :return:
    """
    # Get a mapping from stations to their variable (t2m, slp) and y,x coords in matrix
    station_indx_map = get_station_indices_map()

    # Read Test data and crop to base window:
    orig_test_set = np.load(str(SRC_DIR) + ORIG_TEST_SET)
    test_set = orig_test_set[:,
               BASE_WINDOW_Y_CROP_1:BASE_WINDOW_Y_CROP_2,
               BASE_WINDOW_X_CROP_1:BASE_WINDOW_X_CROP_2]

    np.save(TARGET_DATASETS_DIR + "ground_truth.npy", test_set)

    obs_stations = extract_stations_from_npy(test_set.copy(), station_indx_map)  # Already normalized...
    obs_dates = pd.date_range(start="1950-01-01", periods=test_set.shape[0], freq="D")

    ap_stations = get_analog_pool_stations(station_indx_map)
    ap_dates = pd.date_range(start="1965-01-01", periods=len(list(ap_stations.values())[0]), freq="D")

    print("Get IDs of analog candidates")
    apw_ids = get_analog_pool_windowed_ids(obs_dates, ap_dates, WINDOW_SIDE_SIZE)

    obs_df = pd.DataFrame(data=obs_stations)
    ap_df = pd.DataFrame(data=ap_stations)

    print("Create observation candidate day pairs")
    observations_candidates = get_observation_candidates_pairs(obs_df, ap_df, apw_ids)

    # Create ARM and ARM-WT data in parallel! UGLY but works....

    print("Get nearest analogue neighbor WITH restriction on weather types")
    test_weather_types = pd.read_csv(str(SRC_DIR) + METADATA_DIR + "test_weather_types.csv")
    analog_weather_types = pd.read_csv(str(SRC_DIR) + METADATA_DIR + "analog_pool_weather_types.csv")
    analog_ids_wt, dists_wt = get_analog_ids_with_same_weather_types(observations_candidates, apw_ids,
                                                                     obs_wts=test_weather_types,
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
    analog_pool = get_analog_pool()
    analog_ids = analog_df['analog_ids'].values
    analog_ids_wt = analog_df_wt['analog_ids'].values

    analog_day_set = analog_pool[analog_ids]
    analog_day_set_wt = analog_pool[analog_ids_wt]

    # Load Test Mask and crop to base window
    whole_area_mask = np.load("../masks/mnar/test_mask_99p.npy")
    mask = whole_area_mask[:, BASE_WINDOW_Y_CROP_1:BASE_WINDOW_Y_CROP_2, BASE_WINDOW_X_CROP_1:BASE_WINDOW_X_CROP_2]

    # Get locations (cells) that are not yet observed, to sample from.
    lats = list(range(H))
    lons = list(range(W))
    all_y_x_pairs = list(product(lats, lons))

    ta_obs_locs = list(map(lambda y, x: (y, x), *np.where(mask[0, :, :, 0] == 0)))
    slp_obs_locs = list(map(lambda y, x: (y, x), *np.where(mask[0, :, :, 1] == 0)))

    ta_locs_sample_space = list(set(all_y_x_pairs) - set(ta_obs_locs))
    slp_locs_sample_space = list(set(all_y_x_pairs) - set(slp_obs_locs))

    # Random sample locations and represent them in a mask
    addon_mask = np.ones_like(mask)
    nbr_spatial_cells = addon_mask.shape[1] * addon_mask.shape[2]
    samples_to_draw = int(0.03 * nbr_spatial_cells)  # sample 3%
    print("Number of samples to draw: ", samples_to_draw)

    # For each day, sample additional cells for both variables randomly
    for i in range(addon_mask.shape[0]):
        ta_day_msk_ids = random.choices(ta_locs_sample_space, k=samples_to_draw)
        slp_day_msk_ids = random.choices(slp_locs_sample_space, k=samples_to_draw)

        ta_ys = [loc[0] for loc in ta_day_msk_ids]
        ta_xs = [loc[1] for loc in ta_day_msk_ids]
        slp_ys = [loc[0] for loc in slp_day_msk_ids]
        slp_xs = [loc[1] for loc in slp_day_msk_ids]

        # Set them as observed.
        addon_mask[i, ta_ys, ta_xs, 0] = 0
        addon_mask[i, slp_ys, slp_xs, 1] = 0

    masked_analogs = np.ma.masked_array(analog_day_set, mask=addon_mask)
    masked_analogs_wt = np.ma.masked_array(analog_day_set_wt, mask=addon_mask)
    print(f"Addon mask contains {100 * (1 - addon_mask.mean()):.2f}% added observations.")

    # Apply addon mask and analog variables to test data
    mask_arm_enhanced = mask.copy()
    mask_arm_enhanced_wt = mask.copy()

    mask_arm_enhanced[~masked_analogs.mask] = masked_analogs.mask[~masked_analogs.mask]
    mask_arm_enhanced_wt[~masked_analogs_wt.mask] = masked_analogs_wt.mask[~masked_analogs_wt.mask]
    print(f"Resulting missing rate: {np.mean(mask_arm_enhanced)}")

    variables = test_set.copy()
    variables_wt = test_set.copy()
    replace_val = np.array([T2M_REPLACE_VALUE_B, SLP_REPLACE_VALUE_B])
    masked_mat = np.ma.masked_array(variables, mask=mask)
    masked_mat_wt = np.ma.masked_array(variables_wt, mask=mask)

    # Replace masked values with mean value of corresponding variable
    masked_mat = masked_mat.astype(np.float64).filled(replace_val)
    masked_mat_wt = masked_mat_wt.astype(np.float64).filled(replace_val)

    variables_arm_enhanced = np.ma.getdata(masked_mat)
    variables_arm_enhanced_wt = np.ma.getdata(masked_mat_wt)

    variables_arm_enhanced[~masked_analogs.mask] = masked_analogs.data[~masked_analogs.mask]
    variables_arm_enhanced_wt[~masked_analogs_wt.mask] = masked_analogs_wt.data[~masked_analogs_wt.mask]

    # Save arm enhanced mask and variables
    np.save(f"arm_data/mask_arm.npy", mask_arm_enhanced)
    np.save(f"arm_data/mask_arm_WT.npy", mask_arm_enhanced_wt)  # This mask is the same as mask_amr....
    np.save(f"arm_data/variables_arm.npy", variables_arm_enhanced)
    np.save(f"arm_data/variables_arm_WT.npy", variables_arm_enhanced_wt)


if __name__ == "__main__":
    main()
