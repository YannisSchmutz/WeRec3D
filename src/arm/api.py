import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.config import DATA_DIR_B, TRAIN_SET_NAME
from src.config import BASE_WINDOW_Y_CROP_1, BASE_WINDOW_Y_CROP_2, BASE_WINDOW_X_CROP_1, BASE_WINDOW_X_CROP_2

import os
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath("__init__.py"))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()


def extract_stations_from_npy(data_mat, stat_indx_map):
    """
    Returns dict of only station time series of a corresponding numpy data set.
    :param data_mat: Numpy dataset (e.g. test data GT npy)
    :param stat_indx_map: Dataframe referencing the station locations.
    :return:
    """
    stations = {}
    for index in stat_indx_map.index:
        station_id = stat_indx_map.loc[index].id
        variable = stat_indx_map.loc[index].variable
        y = stat_indx_map.loc[index].y
        x = stat_indx_map.loc[index].x

        var_dim = 0 if variable == "ta" else 1

        # just time dim.
        station_val = data_mat[:, y, x, var_dim]
        stations[f"{station_id}_{variable}"] = station_val
    return stations


def get_analog_pool():
    analog_pool = np.load(str(SRC_DIR) + DATA_DIR_B + TRAIN_SET_NAME)
    analog_pool = analog_pool[:, BASE_WINDOW_Y_CROP_1:BASE_WINDOW_Y_CROP_2, BASE_WINDOW_X_CROP_1:BASE_WINDOW_X_CROP_2]
    return analog_pool


def get_analog_pool_stations(station_indx_map):
    analog_pool = get_analog_pool()
    ap_stations = extract_stations_from_npy(analog_pool, station_indx_map)
    return ap_stations


def get_analog_pool_windowed_ids(obs_dates, ap_dates, window_side_len):
    analog_pool_windowed_ids = []
    for obs_date in obs_dates:
        # Find all objects with same day and month
        obs_day = obs_date.day
        obs_month = obs_date.month

        same_day_ids = np.squeeze(np.argwhere(ap_dates.day == obs_day), axis=-1)
        same_month_ids = np.squeeze(np.argwhere(ap_dates.month == obs_month), axis=-1)
        same_calendar_day_ids = np.intersect1d(same_day_ids, same_month_ids, assume_unique=True)
        windowed_ids = np.array([np.arange(i-window_side_len, i+window_side_len+1) for i in same_calendar_day_ids]).flatten()
        # Remove IDs that are not in ap_dates
        windowed_ids = windowed_ids[windowed_ids >= 0]
        windowed_ids = windowed_ids[windowed_ids < len(ap_dates)]
        analog_pool_windowed_ids.append(windowed_ids)
    return analog_pool_windowed_ids



def get_observation_candidates_pairs(_obs_df, _ap_df, window_ids):

    obs_can_pairs = []
    for t in range(_obs_df.shape[0]):
        obs_vector = _obs_df.loc[t].dropna().values
        observed_stations = _obs_df.loc[t].dropna().index  # Index is columns here

        analog_candidates = _ap_df[observed_stations].loc[window_ids[t]].values
        obs_can_pairs.append((obs_vector, analog_candidates))

    return obs_can_pairs


def get_analog_ids(observations_candidates, window_ids):
    analog_ids = []
    analog_distances = []
    for t, (obs, X) in enumerate(observations_candidates):
        # Distance: Default is “minkowski”, which results in the standard Euclidean distance
        model = KNeighborsClassifier(n_neighbors=1,
                                     algorithm='brute').fit(X,
                                                            np.zeros((X.shape[0]))  # Pseudo labels...
                                                            )
        distances, indices = model.kneighbors([obs])
        print(f"day_{t}: dist={distances[0][0]}")
        candidate_id = indices[0][0]
        analog_id = window_ids[t][candidate_id]

        analog_ids.append(analog_id)
        analog_distances.append(distances[0][0])
    return analog_ids, analog_distances


def get_analog_ids_with_same_weather_types(observations_candidates, window_ids, obs_wts, analog_wts):
    analog_ids = []
    analog_distances = []
    for t, (obs_t, candidates_t) in enumerate(observations_candidates):
        ids_day_t = window_ids[t]
        weather_types_obs_t = obs_wts.loc[t][obs_wts.columns[1:]].values
        weather_types_candidates_t = analog_wts.loc[ids_day_t]['wt']

        same_wt_selector = weather_types_candidates_t.isin(weather_types_obs_t)
        same_wt_candidates = candidates_t[same_wt_selector]
        same_wt_ids = ids_day_t[same_wt_selector]

        # Distance: Default is “minkowski”, which results in the standard Euclidean distance
        model = KNeighborsClassifier(n_neighbors=1,
                                     algorithm='brute').fit(same_wt_candidates,
                                                            np.zeros((same_wt_candidates.shape[0]))  # Pseudo labels...
                                                            )
        distances, indices = model.kneighbors([obs_t])
        print(f"day_{t}: dist={distances[0][0]}")
        candidate_id = indices[0][0]
        analog_id = same_wt_ids[candidate_id]
        analog_ids.append(analog_id)
        analog_distances.append(distances[0][0])
    return analog_ids, analog_distances
