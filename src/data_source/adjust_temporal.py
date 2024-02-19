import xarray as xr
import numpy as np
import os
import statsmodels.api as sm
import subprocess

SOURCE_DIR = "reduced_size_1dg/"
TARGET_DIR = "temporal_adjusted/"
LAND_SEA_MASK = "land_sea_masks/land_sea_mask.nc"
OFFSET = "offsets/EKF400_offset_era5_1grad.nc"

NAME_KEY_MAP = {"2m_temperature": "t2m",
                "mean_sea_level_pressure": "msl"}
KEY_NAME_MAP = {v: k for k, v in NAME_KEY_MAP.items()}


def get_ds(key):
    name = KEY_NAME_MAP[key]
    print(f"    [*] Creating whole DS for {name}")

    files = sorted(list(filter(lambda f: name in f, os.listdir(SOURCE_DIR))))
    print(f"    [*] Reading {len(files)}")
    print(f"    [*] First: {files[0]}, last: {files[-1]}")

    data_sets = []
    for f in files:
        ds = xr.open_dataset(SOURCE_DIR + f)
        data_sets.append(ds)

    ds = xr.concat(data_sets, dim='time')
    return ds


def get_land_sea_mask():
    lsm = xr.open_dataset(LAND_SEA_MASK)
    mask_2d = 1 - lsm['lsm'].to_numpy()
    return mask_2d


def calculate_masked_zonal_means(ds, mask_2d):
    mat = ds['t2m'].to_numpy()
    mask_3d = np.repeat(np.expand_dims(mask_2d, axis=0), mat.shape[0], axis=0)
    masked_mat = np.ma.masked_array(mat, mask=mask_3d)
    masked_zonal_mean = np.mean(masked_mat, axis=2).data
    return masked_zonal_mean


def model_zonal_means(mz_means):
    models = []
    x = list(range(mz_means.shape[0]))
    x_sm = sm.add_constant(x)

    for i in range(mz_means.shape[1]):
        y = mz_means[:, i]
        model = sm.OLS(y, x_sm).fit()
        models.append(model)
    return models


def get_trend_subtraction_mat(models, nbr_longitudes):
    centred_lat_subtractors = []
    for lat_id in range(len(models)):
        cent_lat_sub = models[lat_id].fittedvalues - models[lat_id].fittedvalues[len(models[lat_id].fittedvalues) // 2]
        centred_lat_subtractors.append(cent_lat_sub)

    # Change to [time, latitude]
    centred_lat_subtractors = np.array(centred_lat_subtractors).T

    # Repeat zonal mean over the longitudes
    centred_lat_subtractors = np.expand_dims(centred_lat_subtractors, axis=-1)
    centred_lat_subtractors = np.repeat(centred_lat_subtractors,
                                        repeats=nbr_longitudes, axis=-1)

    return centred_lat_subtractors


def get_offset_mat(nbr_days):
    offset = xr.open_dataset(OFFSET)
    offset_mat = offset['EKF400 offset'].to_numpy()[::-1]
    offset_mat = np.expand_dims(offset_mat, axis=0)
    offset_mat = np.repeat(offset_mat, repeats=nbr_days, axis=0)
    return offset_mat


def create_t2m_deseasonalized(trend_offset_filename):

    # Create ydaymean file
    subprocess.run(["cdo", "ydaymean", TARGET_DIR + trend_offset_filename,
                    TARGET_DIR + "era5_t2m_ydaymean_detrend_offset_19502022.nc"])

    # Create deseasonalized file
    subprocess.run(["cdo", "-b", "F32", "ydaysub",
                    TARGET_DIR + trend_offset_filename,
                    TARGET_DIR + "era5_t2m_ydaymean_detrend_offset_19502022.nc",
                    TARGET_DIR + "era5_t2m_daymean_detrend_offset_deseas_19502022.nc"])

def create_ds_from_mat(mat, coords):
    da = xr.DataArray(mat,
                      dims=("time", "latitude", "longitude"),
                      coords={"time": coords["time"],
                              "longitude": coords["longitude"],
                              "latitude": coords["latitude"]})

    return da.to_dataset(name="t2m")


def process_t2m():
    print(f"[*] Processing t2m...")

    ds = get_ds("t2m")
    nbr_longs = len(ds.coords['longitude'])
    nbr_days = len(ds.coords['time'])

    print(f"[*] Read and prepare land sea mask")
    lsm = get_land_sea_mask()
    print(f"[*] Calculate zonal means")
    msk_zonal_means = calculate_masked_zonal_means(ds, lsm)
    print(f"[*] Model trend")
    mzm_models = model_zonal_means(msk_zonal_means)
    trend_sub_mat = get_trend_subtraction_mat(mzm_models, nbr_longitudes=nbr_longs)
    print(f"[*] Remove trend")
    detrended = ds['t2m'].to_numpy() - trend_sub_mat

    print(f"[*] Read and prepare offset")
    offset_mat = get_offset_mat(nbr_days)
    print(f"[*] Remove offset")
    trend_offset_adjusted = detrended - offset_mat

    toa_ds_file_name = "era5_t2m_daymean_detrend_offset_19502022.nc"
    print(f"[*] Store detrended t2m {TARGET_DIR + toa_ds_file_name}")

    toa_ds = create_ds_from_mat(trend_offset_adjusted, ds.coords)
    toa_ds.to_netcdf(TARGET_DIR + toa_ds_file_name)

    print(f"[*] Create and store anomalie t2m")
    create_t2m_deseasonalized(toa_ds_file_name)
    print("[*] Done!")


def proces_msl():
    print(f"[*] Processing msl...")
    ds = get_ds("msl")
    ds_file_name = "era5_msl_daymean_19502022.nc"
    ds.to_netcdf(TARGET_DIR + ds_file_name)

    print(f"[*] Extract anomalie")
    # Extract anomalie
    # Create ydaymean file
    subprocess.run(["cdo", "ydaymean", TARGET_DIR + ds_file_name,
                    TARGET_DIR + "era5_msl_ydaymean_19502022.nc"])

    # Create anomalie file
    subprocess.run(["cdo", "-b", "F32", "ydaysub",
                    TARGET_DIR + ds_file_name,
                    TARGET_DIR + "era5_msl_ydaymean_19502022.nc",
                    TARGET_DIR + "era5_msl_daymean_deseas_19502022.nc"])
    print("[*] Done!")


if __name__ == '__main__':
    process_t2m()
    proces_msl()
