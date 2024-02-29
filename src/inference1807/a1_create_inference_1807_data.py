
"""
Arrangement Script 1
====================
Creates:
* Inference 1807 GT (.nc)
* Variables 1807 (.npy)
* Mask 1807 (.npy)
"""

import pandas as pd
import pyreadr
import numpy as np
import xarray as xr

import os, sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath("__init__.py"))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()
sys.path.append(os.path.dirname(SRC_DIR))


from src.config import TA_MU, TA_SIGMA, SLP_MU, SLP_SIGMA
from src.config import T2M_REPLACE_VALUE_B, SLP_REPLACE_VALUE_B

from src.config import H as BASE_HEIGHT
from src.config import W as BASE_WIDTH

from arrangement_script_helpers import add_matrix_ids_to_coords
from data_transformer import standardize_data
from data_provider import get_station_indices_map

TARGET_DATASETS_DIR = "data_sets/"
TARGET_MASK_DIR = "masks"


def prepare_bias_df(diff_df, repeat_days):
    diff_t = diff_df.drop(['lon', 'lat'], axis=1).T
    diff_t = diff_t.rename(columns={i:diff_t.loc['id'].values[i] for i in diff_t.columns})
    diff_t = diff_t.drop('id', axis=0)

    df_to_subtract = pd.DataFrame(np.repeat(diff_t.values, repeat_days, axis=0))
    df_to_subtract = df_to_subtract.rename(columns={i:diff_t.columns[i] for i in df_to_subtract.columns})

    return df_to_subtract


def create_ground_truth(ta_df, slp_df, coord_df, dates):
    datasets = []
    for var_df in (ta_df, slp_df):
        for col in var_df.columns:
            station_id, var_name = col.split("_")
            # print(station_id, var_name)
            lat = coord_df[((coord_df['id'] == station_id) & (coord_df['variable'] == var_name))]['lat'].values[0]
            lon = coord_df[((coord_df['id'] == station_id) & (coord_df['variable'] == var_name))]['lon'].values[0]

            var_val = var_df[col].to_numpy()
            var_val = np.expand_dims(np.expand_dims(var_val, axis=0), axis=0)

            ds = xr.Dataset(data_vars={var_name: (["lat", "lon", "time"], var_val)},
                            coords=dict(lat=[lat], lon=[lon], time=dates)
                            )
            datasets.append(ds)

    return xr.merge(datasets)


def create_3d_representation(standardized_var, coords, missing_repr_val):
    mask_mat = np.ones([standardized_var.shape[0], BASE_HEIGHT, BASE_WIDTH])
    var_mat = np.zeros([standardized_var.shape[0], BASE_HEIGHT, BASE_WIDTH])
    var_mat += missing_repr_val

    for loc_id in standardized_var.columns:
        x, y = coords.loc[loc_id][['x', 'y']].astype('int')
        observed_day_ids = [i for i in range(standardized_var.shape[0]) if not standardized_var[loc_id].isna().values[i]]

        var_mat[observed_day_ids, y, x] = standardized_var[loc_id].loc[observed_day_ids].values
        mask_mat[observed_day_ids, y, x] = 0
    return var_mat, mask_mat


def main():
    # Read Bias data
    diff_ta = pd.read_csv("bias_data/diff_station_raster_ta.txt", delimiter=" ")
    diff_slp = pd.read_csv("bias_data/diff_station_raster_slp.txt", delimiter=" ")

    # Store station coordinates
    ta_coords = add_matrix_ids_to_coords(diff_ta[['id', 'lat', 'lon']])
    slp_coords = add_matrix_ids_to_coords(diff_slp[['id', 'lat', 'lon']])

    # Read Station data
    stations = pyreadr.read_r('station_data/analogue_stationdata_detrended_2023-03-30 1.RData')
    stations = stations['TOT']
    stations['date'] = pd.to_datetime(stations['date'])
    stations1807 = stations[stations['date'].dt.year == 1807]
    # Drop unused columns
    stations1807 = stations1807.drop(['WT_type1', 'WT_probability1', 'WT_type2', 'WT_probability2',
                                      'WT_type3', 'WT_probability3', 'WT_type4', 'WT_probability4',
                                      'WT_type5', 'WT_probability5', 'WT_type6', 'WT_probability6',
                                      'WT_type7', 'WT_probability7'], axis=1)

    # Split per variables
    ta_stations = stations1807[[col for col in stations1807.columns if "_ta" in col]]
    slp_stations = stations1807[[col for col in stations1807.columns if "_slp" in col]]

    days_per_month = stations1807.groupby(stations1807['date'].dt.month).count()['date'].values

    ta_to_subtract = prepare_bias_df(diff_ta, days_per_month)
    slp_to_subtract = prepare_bias_df(diff_slp, days_per_month)

    debiased_ta = ta_stations.subtract(ta_to_subtract)
    debiased_slp = slp_stations.subtract(slp_to_subtract)
    # Convert from hPa to Pa
    debiased_slp = debiased_slp * 100

    # Create GT values as nc
    station_coordinate_df = get_station_indices_map()
    ground_truth = create_ground_truth(debiased_ta, debiased_slp, station_coordinate_df, stations1807['date'])
    ground_truth.to_netcdf(f"{TARGET_DATASETS_DIR}/ground_truth.nc")

    # Preprocessing for DL-modelling
    standardized_ta = standardize_data(debiased_ta, TA_MU, TA_SIGMA)
    standardized_slp = standardize_data(debiased_slp, SLP_MU, SLP_SIGMA)

    mat_ta, mask_ta = create_3d_representation(standardized_ta, ta_coords, T2M_REPLACE_VALUE_B)
    mat_slp, mask_slp = create_3d_representation(standardized_slp, slp_coords, SLP_REPLACE_VALUE_B)

    mat = np.concatenate([np.expand_dims(mat_ta, axis=-1),
                          np.expand_dims(mat_slp, axis=-1)
                          ], axis=-1)

    mask = np.concatenate([np.expand_dims(mask_ta, axis=-1),
                           np.expand_dims(mask_slp, axis=-1)
                           ], axis=-1)

    np.save(f'{TARGET_DATASETS_DIR}/variables_set.npy', mat)
    np.save(f'{TARGET_MASK_DIR}/mask_set.npy', mask)


if __name__ == '__main__':
    main()
