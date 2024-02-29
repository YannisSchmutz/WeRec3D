import os
import sys
from pathlib import Path
SCRIPT_DIR = os.path.dirname(os.path.abspath("__init__.py"))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()
print(SRC_DIR)
sys.path.append(os.path.dirname(SRC_DIR))
sys.path.append(os.path.dirname(str(SRC_DIR) + '/models'))

import click
import numpy as np
import pandas as pd

from data_provider import get_inference_x_data, get_station_indices_map
from modelling_helper import load_model, full_prediction, loo_prediction
from data_transformer import scale_back


def run_full(missing_like):
    x = get_inference_x_data(missing_like)
    model = load_model(missing_like)
    pred = full_prediction(model, x)
    scaled_pred = scale_back(pred)
    np.save(f"predictions/full_pred_{missing_like}.npy", scaled_pred)


def run_loo(missing_like):
    dates = pd.date_range(start="1807-01-01", periods=365, freq="D")

    model = load_model(missing_like)
    x = get_inference_x_data(missing_like)
    station_indices_map = get_station_indices_map()
    loo_pred = loo_prediction(model, x, dates, station_indices_map)  # Already scaled back!
    loo_pred.to_netcdf(f"predictions/loo_pred_{missing_like}.nc")


@click.command()
@click.option('-r', '--run_type', required=True, type=str)
@click.option('-m', '--missing_like', required=True, type=str)
def main(run_type, missing_like):
    """
    Run commands:
    python reconstruct_historic_weather.py -r full -m plain
    python reconstruct_historic_weather.py -r full -m arm
    python reconstruct_historic_weather.py -r full -m arm_wt
    python reconstruct_historic_weather.py -r loo -m plain
    python reconstruct_historic_weather.py -r loo -m arm
    python reconstruct_historic_weather.py -r loo -m arm_wt

    :param run_type: full / loo
    :param missing_like: plain / arm / arm_wt
    :return:
    """
    if missing_like not in ("plain", "arm", "arm_wt"):
        raise NotImplementedError(f"Your missing_like={missing_like} is not supported")

    if run_type == "full":
        run_full(missing_like)
    elif run_type == "loo":
        run_loo(missing_like)
    else:
        raise NotImplementedError(f"Your run_type={run_type} is not supported")


if __name__ == "__main__":
    main()
