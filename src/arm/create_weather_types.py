import pandas as pd
from collections import defaultdict

import os, sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath("__init__.py"))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()
sys.path.append(os.path.dirname(SRC_DIR))

from src.config import METADATA_DIR
from src.config import WT_PROBABILITY_THRESHOLD


def get_high_probability_wts(df):
    high_probability_wts = defaultdict(list)

    for _id, row in df.iterrows():
        tmp_probab = 0
        date = row.date.strftime("%Y-%m-%d")
        for i in range(1, len(row), 2):
            type_i = row[i]
            probab_i = row[i+1]
            if tmp_probab < WT_PROBABILITY_THRESHOLD:
                high_probability_wts[date].append(type_i)
                tmp_probab += probab_i
            else:
                continue
    return dict(high_probability_wts)


def create_high_probab_wt_df(wts):
    max_wt_len = max([len(wt) for wt in list(wts.values())])
    filled_wts = {}
    for k, v in wts.items():
        filled_v = [v[i] if i < len(v) else -1 for i in range(max_wt_len)]
        filled_wts[k] = filled_v

    df = pd.DataFrame(data=filled_wts.values(), index=filled_wts.keys(), columns=[f"wt{i}" for i in range(max_wt_len)])
    return df


def main():
    """
    Create weather types for test, inference and analog pool
    :return:
    """
    df_all = pd.read_csv("wt_classes/CAP7_1763-2009_AllTypes.txt", delimiter="\s+")
    df_all['date'] = pd.to_datetime(df_all['date'])

    df_test = df_all[(df_all.date.dt.year >= 1950) & (df_all.date.dt.year <= 1954)]
    df_inf = df_all[df_all.date.dt.year == 1807]

    test_wts = get_high_probability_wts(df_test)
    inf_wts = get_high_probability_wts(df_inf)

    test_wt_df = create_high_probab_wt_df(test_wts)
    inf_wt_df = create_high_probab_wt_df(inf_wts)

    test_wt_df.to_csv(str(SRC_DIR) + METADATA_DIR + "test_weather_types.csv", index_label="date")
    inf_wt_df.to_csv(str(SRC_DIR) + METADATA_DIR + "inference1807_weather_types.csv", index_label="date")

    # Read WTs for analog pool
    df_meteo = pd.read_csv("wt_classes/CAP9_MeteoSchweiz_1957-2022.csv")
    df_meteo['time'] = pd.to_datetime(df_meteo['time'])
    df_analog_pool_wt = df_meteo[(df_meteo.time.dt.year >= 1965) & (df_meteo.time.dt.year <= 2020)]
    df_analog_pool_wt.columns = ['date', 'wt']
    df_analog_pool_wt = df_analog_pool_wt.set_index('date')

    df_analog_pool_wt.to_csv(str(SRC_DIR) + METADATA_DIR + "analog_pool_weather_types.csv", index_label="date")


if __name__ == "__main__":
    main()
