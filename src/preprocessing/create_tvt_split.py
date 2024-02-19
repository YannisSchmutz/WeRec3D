import xarray as xr
import json
import datetime
import numpy as np
import pandas as pd


SRC_DIR = "../data_source/temporal_adjusted/"
VARIANT_A_T2M = "era5_t2m_daymean_detrend_offset_deseas_19502022.nc"
VARIANT_A_MSL = "era5_msl_daymean_deseas_19502022.nc"

VARIANT_B_T2M = "era5_t2m_daymean_detrend_offset_19502022.nc"
VARIANT_B_MSL = "era5_msl_daymean_19502022.nc"


# Detrended and deseasonalized
DST_DIR_VARIANT_A = "../data_sets/variant_a/"
# Just detrended
DST_DIR_VARIANT_B = "../data_sets/variant_b/"


TRAIN_DATE_RANGE = ("1965-01-01", "2020-12-31")
VAL_DATE_RANGE = ("1955-01-01", "1964-12-31")
TEST_DATE_RANGE = ("1950-01-01", "1954-12-31")


def _write_metadata(file_name_path, metadata):
    metadata['created_at'] = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    with open(file_name_path, "w") as fh:
        fh.write(json.dumps(metadata, indent=2))


def _get_tvt_split(xds, variable_name):
    xds["time"] = xds["time"].dt.floor("D")

    train_split_start = TRAIN_DATE_RANGE[0]
    train_split_end = TRAIN_DATE_RANGE[1]
    validation_split_start = VAL_DATE_RANGE[0]
    validation_split_end = VAL_DATE_RANGE[1]
    test_split_start = TEST_DATE_RANGE[0]
    test_split_end = TEST_DATE_RANGE[1]

    train_set_time_range = pd.date_range(start=train_split_start, end=train_split_end, freq="D")
    validation_set_time_range = pd.date_range(start=validation_split_start, end=validation_split_end, freq="D")
    test_set_time_range = pd.date_range(start=test_split_start, end=test_split_end, freq="D")

    print(f"Create {variable_name} train split")
    train_matrix = xds.sel(dict(time=train_set_time_range))[variable_name].to_numpy()
    print(f"Create {variable_name} validation split")
    validation_matrix = xds.sel(dict(time=validation_set_time_range))[variable_name].to_numpy()
    print(f"Create {variable_name} test split")
    test_matrix = xds.sel(dict(time=test_set_time_range))[variable_name].to_numpy()
    return train_matrix.astype('float32'), validation_matrix.astype('float32'), test_matrix.astype('float32')


def _get_standardization_parameter(variable_matrix):
    _mean = np.mean(variable_matrix)
    _std = np.std(variable_matrix)
    return _mean.astype('float32'), _std.astype('float32')


def _standardize_data(dataset_matrix, mu, sigma):
    return (dataset_matrix - mu) / sigma


def create_tvt_split(*, t2m_file, msl_file, dst_dir):

    metadata = {"created_at": "",
                "scaling_parameters": {}
                }

    ds_t2m = xr.open_dataset(SRC_DIR + t2m_file)
    ds_msl = xr.open_dataset(SRC_DIR + msl_file)

    # Temporal train, validation test split
    t2m_train, t2m_validation, t2m_test = _get_tvt_split(ds_t2m, variable_name='t2m')
    del ds_t2m  # free memory
    msl_train, msl_validation, msl_test = _get_tvt_split(ds_msl, variable_name='msl')
    del ds_msl  # free memory

    t2m_mean, t2m_std = _get_standardization_parameter(t2m_train)
    msl_mean, msl_std = _get_standardization_parameter(msl_train)

    metadata["scaling_parameters"]["t2m"] = {"mu": str(t2m_mean), "sigma": str(t2m_std)}
    metadata["scaling_parameters"]["msl"] = {"mu": str(msl_mean), "sigma": str(msl_std)}

    print("Scale t2m train data")
    t2m_train = _standardize_data(t2m_train, mu=t2m_mean, sigma=t2m_std)
    print("Scale t2m validation data")
    t2m_validation = _standardize_data(t2m_validation, mu=t2m_mean, sigma=t2m_std)
    print("Scale t2m test data")
    t2m_test = _standardize_data(t2m_test, mu=t2m_mean, sigma=t2m_std)

    print("Scale msl train data")
    msl_train = _standardize_data(msl_train, mu=msl_mean, sigma=msl_std)
    print("Scale msl validation data")
    msl_validation = _standardize_data(msl_validation, mu=msl_mean, sigma=msl_std)
    print("Scale msl test data")
    msl_test = _standardize_data(msl_test, mu=msl_mean, sigma=msl_std)

    print("Create train set")
    train_set = np.stack((t2m_train, msl_train), axis=-1)
    print(f"Train set shape: {train_set.shape}")
    del t2m_train, msl_train

    print("Create validation set")
    validation_set = np.stack((t2m_validation, msl_validation), axis=-1)
    print(f"Validation set shape: {validation_set.shape}")
    del t2m_validation, msl_validation

    print("Create test set")
    test_set = np.stack((t2m_test, msl_test), axis=-1)
    print(f"Test set shape: {test_set.shape}")

    # SAVE
    np.save(dst_dir + "train.npy", train_set.astype('float32'))
    np.save(dst_dir + "validation.npy", validation_set.astype('float32'))
    np.save(dst_dir + "test.npy", test_set.astype('float32'))

    print(metadata)
    meta_data_file_name_path = dst_dir + "metadata.json"
    _write_metadata(meta_data_file_name_path, metadata)

    print("DONE")


if __name__ == "__main__":
    print("Create variant A")
    create_tvt_split(t2m_file=VARIANT_A_T2M,
                     msl_file=VARIANT_A_MSL,
                     dst_dir=DST_DIR_VARIANT_A)

    print("Create variant B")
    create_tvt_split(t2m_file=VARIANT_B_T2M,
                     msl_file=VARIANT_B_MSL,
                     dst_dir=DST_DIR_VARIANT_B)
