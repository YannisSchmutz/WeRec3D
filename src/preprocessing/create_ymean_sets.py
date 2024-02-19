import xarray as xr
import numpy as np
import pandas as pd


Y_DAY_MEAN_DIR = "../data_source/temporal_adjusted/"
Y_DAY_MEAN_T2M = "era5_t2m_ydaymean_detrend_offset_19502022.nc"
Y_DAY_MAEN_MSL = "era5_msl_ydaymean_19502022.nc"

VAL_DATE_RANGE = ("1955-01-01", "1964-12-31")
TEST_DATE_RANGE = ("1950-01-01", "1954-12-31")

DST_DIR = "../data_sets/ymean_sets/"


def create_ydaymean_matrix(file_name, var):
    ydaymean_mat = xr.open_dataset(Y_DAY_MEAN_DIR + file_name)
    ydaymean_mat = ydaymean_mat[var].to_numpy()
    ydaymean_mat = ydaymean_mat[:, 6:38, 2:66]
    return ydaymean_mat


def create_data_set_ymean(dates, t2m_ymean, msl_ymean):
    ymean_set = np.zeros((len(dates), 32, 64, 2))
    doy_ids = [vd.day_of_year - 1 for vd in dates]

    ymean_set[..., 0] = t2m_ymean[doy_ids]
    ymean_set[..., 1] = msl_ymean[doy_ids]
    return ymean_set


def main():
    # Create numpy matrices of base window
    t2m_ymean = create_ydaymean_matrix(Y_DAY_MEAN_T2M, 't2m')
    msl_ymean = create_ydaymean_matrix(Y_DAY_MAEN_MSL, 'msl')

    val_dates = pd.date_range(start=VAL_DATE_RANGE[0], end=VAL_DATE_RANGE[1], freq="D")
    test_dates = pd.date_range(start=TEST_DATE_RANGE[0], end=TEST_DATE_RANGE[1], freq="D")

    val_set_ymean = create_data_set_ymean(val_dates, t2m_ymean, msl_ymean)
    test_set_ymean = create_data_set_ymean(test_dates, t2m_ymean, msl_ymean)

    print("Created val and test ymean sets with shape:")
    print(val_set_ymean.shape)
    print(test_set_ymean.shape)

    # SAVE
    np.save(DST_DIR + "validation_ymean.npy", val_set_ymean.astype('float32'))
    np.save(DST_DIR + "test_ymean.npy", test_set_ymean.astype('float32'))


if __name__ == "__main__":
    main()
