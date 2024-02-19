import xarray as xr
import numpy as np
from skimage.measure import block_reduce
import os

from src.config import LAT_START, LAT_END, LON_START, LON_END

DATA_SOURCE = "daymean_sets/"
DATA_TARGET = "reduced_size_1dg/"


SRC_RESOLUTION = 0.25
TARGET_RESOLUTION = 1
POOLING_FACTOR = int(TARGET_RESOLUTION / SRC_RESOLUTION)


def reduce_spatial_size(ds, *, lon_start=LON_START, lon_end=LON_END, lat_start=LAT_START, lat_end=LAT_END,
                        resolution=SRC_RESOLUTION):
    """
    Crops the field to specified window.
    Default is: (160 x 272)

    :param ds:
    :param lon_start:
    :param lon_end:
    :param lat_start:
    :param lat_end:
    :param resolution:
    :return:
    """
    # np.arange is exclusive on the right side
    return ds.sel(longitude=np.arange(lon_start, lon_end, resolution),
                  latitude=np.arange(lat_start, lat_end, -resolution))


def reduce_spatial_granularity(ds, data_var_key):
    """
    Reduces granularity of given dataset.
        E.g.: (160 x 272) -> (40, 68)
    :param ds:
    :param data_var_key:
    :return:
    """
    ar = ds[data_var_key]
    reduced = block_reduce(ar.to_numpy(),
                           block_size=(1, POOLING_FACTOR, POOLING_FACTOR),
                           func=np.mean)

    reduced_ar = xr.DataArray(reduced,
                              dims=("time", "latitude", "longitude"),
                              coords={"time": ds.coords["time"],
                                      "longitude": np.arange(LON_START, LON_END, TARGET_RESOLUTION),
                                      "latitude": np.arange(LAT_START, LAT_END, -TARGET_RESOLUTION)})

    reduced_ds = reduced_ar.to_dataset(name=data_var_key)
    return reduced_ds


def change_scale(ds):
    """
    Changes scale from kelvin to celsius.
    Only use for t2m!
    :param ds:
    :return:
    """
    ds['t2m'] = ds['t2m'] - 273.15
    return ds


def process_t2m_files():
    source_files = sorted(list(filter(lambda f: "2m_temperature" in f, os.listdir(DATA_SOURCE))))
    for f in source_files:
        if os.path.exists(DATA_TARGET + f):
            print(f"Skip already existing file: {f}")
            continue
        if '2023' in f:
            # Do not include 2023
            continue
        print(f"Processing: {f}")
        ds = xr.open_dataset(DATA_SOURCE + f)
        ds = reduce_spatial_size(ds)
        ds = reduce_spatial_granularity(ds, 't2m')
        ds = change_scale(ds)

        ds.to_netcdf(DATA_TARGET + f)


def process_msl_files():
    source_files = sorted(list(filter(lambda f: "mean_sea_level_pressure" in f, os.listdir(DATA_SOURCE))))
    for f in source_files:
        if os.path.exists(DATA_TARGET + f):
            print(f"Skip already existing file: {f}")
            continue
        if '2023' in f:
            # Do not include 2023
            continue
        print(f"Processing: {f}")
        ds = xr.open_dataset(DATA_SOURCE + f)
        ds = reduce_spatial_size(ds)
        ds = reduce_spatial_granularity(ds, 'msl')
        ds.to_netcdf(DATA_TARGET + f)


if __name__ == '__main__':
    process_t2m_files()
    process_msl_files()
