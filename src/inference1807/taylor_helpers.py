
import numpy as np
import pandas as pd
import statsmodels.api as sm


def get_nan_ids(stations):
    """
    Return for each station the corresponding nan-ids
    :param stations: Extracted stations as dict.
    :return:
    """
    miss_indicies_map = {}
    for station_id, station_vals in stations.items():
        # Remove nan-values in bar.
        missing_indicies = np.argwhere(np.isnan(station_vals))
        if len(missing_indicies):
            missing_indicies = list(np.squeeze(missing_indicies, axis=-1))
        else:
            missing_indicies = []
        miss_indicies_map[station_id] = missing_indicies
    return miss_indicies_map


def extract_anomalies(stations, station_indx_map, miss_id_map=None, n_doy=365):
    """
    Extracts the anomaly of ta and slp. For tha using the first two harmonics, for slp using ymean.
    :param stations: Extracted stations as dict.
    :param station_indx_map: Matrix indices of stations.
    :param miss_id_map: Nan-ids per stations.
    :param n_doy: Number of days in period.
    :return:
    """
    def _extract_for_ta(y_obs, missing_ids=None):
        x_ = np.linspace(0, n_doy - 1, n_doy)  # doy
        x1 = np.sin((2 * np.pi * x_) / n_doy)
        x2 = np.cos((2 * np.pi * x_) / n_doy)
        x3 = np.sin((4 * np.pi * x_) / n_doy)
        x4 = np.cos((4 * np.pi * x_) / n_doy)

        x = pd.DataFrame(data={'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})
        x_sm = sm.add_constant(x)
        if missing_ids:
            x_sm = x_sm.drop(missing_ids)
            y_obs = np.delete(y_obs, missing_ids)
        model = sm.OLS(y_obs, x_sm).fit()
        c0, c1, c2, c3, c4 = model.params
        saisonal_component = c0 + c1 * x_sm.x1 + c2 * x_sm.x2 + c3 * x_sm.x3 + c4 * x_sm.x4
        desaisonalized = y_obs - saisonal_component
        # Insert nans again:
        if missing_ids:
            nan_vals = pd.Series([np.nan for _ in missing_ids], index=missing_ids)
            desaisonalized = pd.concat([desaisonalized, nan_vals], axis=0, ignore_index=False).sort_index()
        return desaisonalized.to_numpy()

    def _get_station_yx(stat_id, station_indx_map):
        stat_abb = stat_id.split('_')[0]
        stat_var = stat_id.split('_')[1]
        yx_pair = \
        station_indx_map.loc[(station_indx_map['id'] == stat_abb) & (station_indx_map['variable'] == stat_var)][
            ['y', 'x']]
        return (yx_pair['y'], yx_pair['x'])

    def _extract_for_slp(stat_vals, stat_id, missing_ids, inference_ymean):
        y, x = _get_station_yx(stat_id, station_indx_map)
        ymean_series = np.squeeze(inference_ymean[:, y, x, 1])
        if missing_ids:
            ymean_series = np.delete(ymean_series, missing_ids)
            stat_vals = pd.Series(stat_vals).drop(missing_ids)
        anomaly = stat_vals - ymean_series
        if missing_ids:
            nan_vals = pd.Series([np.nan for _ in missing_ids], index=missing_ids)
            anomaly = pd.concat([anomaly, nan_vals], axis=0, ignore_index=False).sort_index().to_numpy()
        return anomaly

    inference_ymean = np.load("../data_sets/ymean_sets/inference_ymean.npy")
    handled_stations = {}
    for stat_id, stat_vals in stations.items():
        missing_ids = None
        if miss_id_map:
            missing_ids = miss_id_map[stat_id]
        if stat_id.endswith("_ta"):
            anomaly = _extract_for_ta(stat_vals, missing_ids)
        else:
            anomaly = _extract_for_slp(stat_vals, stat_id, missing_ids, inference_ymean)

        handled_stations[stat_id] = anomaly
    return handled_stations


def get_corr(gt, pred):
    """
    Get correlation between ground truth and prediction.
    :param gt: Numpy-gt
    :param pred: Numpy-pred
    :return:
    """
    mu_r = np.mean(gt)
    mu_f = np.mean(pred)
    sigma_r = np.std(gt)
    sigma_f = np.std(pred)

    diff_r = gt - mu_r
    diff_f = pred - mu_f

    ele_wise_mul = diff_r * diff_f
    numerator = np.mean(ele_wise_mul)
    denominator = sigma_r * sigma_f
    corr = numerator / denominator  # TODO: Division through 0 would break this..!
    return corr


def get_normalized_sigma(vector, reference_sigma):
    """
    Normalizes the standard deviation of a given series by another standard deviation.
    :param vector: Series to get the normalized std.dev from
    :param reference_sigma: Reference std.dev
    :return:
    """
    return np.std(vector) / reference_sigma


def get_normalized_rmse(gt, pred):
    """
    Calculates the normalized rmse.
    :param gt: Ground truth
    :param pred: Prediction
    :return:
    """
    mu_r = np.mean(gt)
    mu_f = np.mean(pred)
    sigma_r = np.std(gt)

    diff_r = gt - mu_r
    diff_f = pred - mu_f

    ele_wise_diff = diff_f - diff_r
    ele_wise_square = np.square(ele_wise_diff)
    numerator = np.sqrt(np.mean(ele_wise_square))
    normed_rmse = numerator / sigma_r
    return normed_rmse


def get_loo_taylor_metrics(gt_stations, pred_stations, missing_indicies):
    """
    Returns a dict of dicts of taylor-metrics for each station.
    :param gt_stations: Extracted GT stations
    :param pred_stations: Extracted Pred stations
    :param missing_indicies: Nan-ids
    :return:
    """
    metrics = {}

    for station_id in gt_stations.keys():
        pred = pred_stations[station_id]
        gt = gt_stations[station_id]

        # Remove unobserved value-indicies of GT from pred-series
        pred_clean = np.delete(pred, missing_indicies[station_id])
        gt_clean = np.delete(gt, missing_indicies[station_id])

        corr = get_corr(gt_clean, pred_clean)
        rmse_hat = get_normalized_rmse(gt_clean, pred_clean)

        # sigma_gt_hat = 1
        sigma_pred_hat = get_normalized_sigma(pred_clean, np.std(gt_clean))

        metrics[station_id] = {"corr": corr, "norm_std": sigma_pred_hat, "norm_rmse": rmse_hat}
    return metrics
