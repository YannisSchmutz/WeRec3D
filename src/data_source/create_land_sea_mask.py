import numpy as np
import xarray as xr
from src.config import LAT_START, LAT_END, LON_START, LON_END

SRC_FILE = 'land_sea_masks/orig_land_sea_mask_1degree.nc'
DST_FILE = 'land_sea_masks/land_sea_mask.nc'

RESOLUTION = 1


def create_lsm(src_file=SRC_FILE):
    lsm = xr.open_dataset(src_file)

    mat = lsm['lsm'].isel(time=0).sel(latitude=np.arange(LAT_START, LAT_END, -RESOLUTION),
                                      longitude=np.concatenate((np.arange(360 + LON_START, 360, RESOLUTION),
                                                                np.arange(0, LON_END, RESOLUTION)))
                                      ).to_numpy()
    mat = mat.round()
    lsm = xr.DataArray(mat,
                       dims=("latitude", "longitude"),
                       coords={"longitude": np.arange(LON_START, LON_END, RESOLUTION),
                               "latitude": np.arange(LAT_START, LAT_END, -RESOLUTION)})
    lsm = lsm.to_dataset(name='lsm')
    lsm.to_netcdf(DST_FILE)


if __name__ == '__main__':
    create_lsm()
