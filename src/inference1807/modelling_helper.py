import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import xarray as xr

SCRIPT_DIR = os.path.dirname(os.path.abspath("__init__.py"))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()
sys.path.append(os.path.dirname(SRC_DIR))

from src.config import MODEL_PATH_PLAIN, MODEL_PATH_ARM_ENHANCED
from src.config import F, H, W, CH, BS
from src.config import T2M_REPLACE_VALUE_B, SLP_REPLACE_VALUE_B
from src.models.model2 import create_model

from data_transformer import get_median_pred_days, scale_back


def load_model(missing_like):
    """
    Returns configured model used for the corresponding missing_like pattern.
    :param missing_like: Str describing missing pattern. E.g. "plain"
    :return:
    """
    if missing_like == "plain":
        model_path = str(SRC_DIR) + MODEL_PATH_PLAIN
    elif missing_like == "arm" or missing_like == "arm_wt":
        model_path = str(SRC_DIR) + MODEL_PATH_ARM_ENHANCED
    else:
        raise NotImplementedError(f"Your missing_like={missing_like} is not supported")

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model(f=F, h=H, w=W, ch=CH, bs=BS)
        model.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly=None)
        model.load_weights(model_path)
    return model


def full_prediction(model, x):
    """
    Full prediction and reshaping to (Days, Height, Width, Channels)
    :param model: Model to predict on.
    :param x: Data to reconstruct.
    :return:
    """
    pred = model.predict(x, batch_size=BS)
    pred = get_median_pred_days(pred)
    return pred



def loo_prediction(model, x, dates, stat_indx_map):
    """
    Returns xarray dataset containing all loo-left-out-time-series.
    :param model: Model to predict on.
    :param x: Data to reconstruct.
    :param dates: Dates to use in the dataset
    :param stat_indx_map: Dataframe referencing the station locations.
    :return:
    """

    datasets = []
    for index in stat_indx_map.index:
        station = stat_indx_map.loc[index].id
        variable = stat_indx_map.loc[index].variable
        y_pos = stat_indx_map.loc[index].y
        x_pos = stat_indx_map.loc[index].x
        lat = stat_indx_map.loc[index].lat
        lon = stat_indx_map.loc[index].lon

        print(f"Leaving out: {station}-{variable} ({index})")

        tmp_x = np.copy(x)
        if variable == "ta":
            var_dim, mask_dim, replace_val = 0, 2, T2M_REPLACE_VALUE_B
        else:
            var_dim, mask_dim, replace_val = 1, 3, SLP_REPLACE_VALUE_B

        # Consider current station position as missing. Here we can just replace across the whole time-dimension.
        # (Samples, Frames, H, W, CH)
        tmp_x[:, :, y_pos, x_pos, var_dim] = replace_val
        tmp_x[:, :, y_pos, x_pos, mask_dim] = 1

        pred = model.predict(x, batch_size=BS)
        pred = get_median_pred_days(pred)

        pred = scale_back(pred)
        station_pred = pred[:, y_pos, x_pos, var_dim]
        station_pred = np.expand_dims(np.expand_dims(station_pred, axis=0), axis=0)

        ds = xr.Dataset(data_vars={variable: (["lat", "lon", "time"], station_pred)},
                        coords=dict(lat=[lat], lon=[lon], time=dates)
                        )
        datasets.append(ds)

    loo_predictions = xr.merge(datasets)
    return loo_predictions
