import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()

# TODO: Change this part! Define it in the Notebooks???
# CHART_PATH = f"{str(SRC_DIR)}/experiments_validation/charts/"

T2M_MU = 7.9444466
T2M_SIGMA = 8.895299
SLP_MU = 101334.484
SLP_SIGMA = 1098.013
VAL_DATE_RANGE = ("1955-01-01", "1964-12-31")
DATES = pd.date_range(start=VAL_DATE_RANGE[0],
                      end=VAL_DATE_RANGE[1],
                      freq="D",
                      ).to_numpy().astype('datetime64[D]')


def visualize_run_learning_curves(run_metrics, save_file_name="", fig_width=16):
    nbr_total_metrics = len(run_metrics.keys())
    nbr_train_val_metrics = nbr_total_metrics // 2

    max_error = max(max(run_metrics.values()))

    fig, axs = plt.subplots(1, nbr_train_val_metrics, figsize=(fig_width, 4))
    for i, (k, v) in enumerate(run_metrics.items()):
        c = "red"
        min_val = None
        min_val_pos = None
        if "val" in k:
            c = "green"
            min_val = min(v)
            min_val_pos = v.index(min_val)

        axs[i//2].plot(v, color=c, label=k)
        axs[i//2].set_ylabel("Error")
        axs[i//2].set_xlabel("Epoch")
        axs[i//2].set_ylim(bottom=0, top=1.1*max_error)
        # Show minimal value
        if min_val:
            axs[i // 2].scatter([min_val_pos], [min_val * 1.08],
                                marker="v", c='black', s=20, label=f"{round(min_val, 3)}")
        axs[i//2].grid(True)
        axs[i//2].legend()
        axs[i//2].set_axisbelow(True)

    if save_file_name:
        plt.savefig(f"{save_file_name}", bbox_inches='tight', pad_inches=0.1)


def visualize_all_runs(runs, metrics, save_file_name=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    max_error = 0

    for run in runs:
        train_label = run.split('_')[-1]
        val_label = f"val_{train_label}"

        train_y = metrics[run]['mae']
        val_y = metrics[run]['val_mae']
        axs[0].plot(train_y, label=train_label)
        axs[1].plot(val_y, label=val_label)

        tmp_max = max([max(train_y), max(val_y)])
        if tmp_max > max_error:
            max_error = tmp_max

    axs[0].set_ylabel("Train MAE [z]")
    axs[1].set_ylabel("Val MAE [z]")

    axs[0].set_xlabel("Epoch")
    axs[1].set_xlabel("Epoch")

    axs[0].grid(True)
    axs[1].grid(True)
    axs[0].legend()
    axs[1].legend()

    axs[0].set_ylim(bottom=0, top=1.1*max_error)
    axs[1].set_ylim(bottom=0, top=1.1*max_error)

    if save_file_name:
        plt.savefig(f"{save_file_name}", bbox_inches='tight', pad_inches=0.1)


def visualize_errors_as_function_of_percentage(runs, metrics, percentages, save_file_name=None):
    x = list(map(lambda p: int(p), percentages))
    min_train_vals = [min(metrics[run]['mae']) for run in runs]
    min_val_vals = [min(metrics[run]['val_mae']) for run in runs]

    plt.scatter(x, min_train_vals, label="Train MAE [z]", c='r')
    plt.scatter(x, min_val_vals, label="Val MAE [z]", c='g')
    plt.gca().legend()
    plt.gca().grid(True)
    plt.xlabel("Percentage Missing")
    plt.ylabel("MAE")

    if save_file_name:
        plt.savefig(f"{save_file_name}", bbox_inches='tight', pad_inches=0.1)


def visualize_qualitative_set(x, y, pred, dates, variable, scale_back=False, save_file_name=None):
    if variable == "t2m":
        vid = 0
    elif variable == "slp":
        vid = 1
    else:
        raise ValueError("Variable must be t2m or slp")
    unit = "z"
    if scale_back:
        unit = "°C" if variable == "t2m" else "Pa"
        mu = T2M_MU if variable == "t2m" else SLP_MU
        sigma = T2M_SIGMA if variable == "t2m" else SLP_SIGMA

        x = x*sigma + mu
        y = y*sigma + mu
        pred = pred*sigma + mu

    fig, axes = plt.subplots(4, 5, figsize=(16, 10))
    fig.subplots_adjust(right=0.95)

    input_vmin = np.min(x[..., vid])
    input_vmax = np.max(x[..., vid])
    pred_vmin = np.min(pred[..., vid])
    pred_vmax = np.max(pred[..., vid])
    gt_vmin = np.min(y[..., vid])
    gt_vmax = np.max(y[..., vid])

    pred_gt_vmin = np.minimum(pred_vmin, gt_vmin)
    pred_gt_vax = np.minimum(pred_vmax, gt_vmax)

    t2m_abs_err = np.abs(pred[..., vid] - y[..., vid])
    error_vmin = np.min(t2m_abs_err)
    error_vmax = np.max(t2m_abs_err)

    for i in range(5):
        # Input
        ms_input = axes[0, i].matshow(x[i, ..., vid], vmin=input_vmin, vmax=input_vmax)
        # t2m-prediction
        ms_pred = axes[1, i].matshow(pred[i, :, :, vid], vmin=pred_gt_vmin, vmax=pred_gt_vax)
        # t2m-ground truth
        ms_gt = axes[2, i].matshow(y[i, :, :, vid], vmin=pred_gt_vmin, vmax=pred_gt_vax)
        # t2m-absolute error
        ms_err = axes[3, i].matshow(t2m_abs_err[i], vmin=error_vmin, vmax=error_vmax, cmap='Reds')

        axes[0, i].set_title(dates[i], fontsize=16)

        if i == 4:
            #                      [left, bottom, width, height]
            sub_ax1 = fig.add_axes([0.96, 0.74, 0.01, 0.11])
            fig.colorbar(ms_input, cax=sub_ax1)
            sub_ax2 = fig.add_axes([0.96, 0.54, 0.01, 0.11])
            fig.colorbar(ms_pred, cax=sub_ax2)
            sub_ax3 = fig.add_axes([0.96, 0.34, 0.01, 0.11])
            fig.colorbar(ms_gt, cax=sub_ax3)
            sub_ax4 = fig.add_axes([0.96, 0.14, 0.01, 0.11])
            fig.colorbar(ms_err, cax=sub_ax4)

    axes[0, 0].set_ylabel(f"Input [{unit}]", fontsize=16)
    axes[1, 0].set_ylabel(f"Pred [{unit}]", fontsize=16)
    axes[2, 0].set_ylabel(f"Target [{unit}]", fontsize=16)
    axes[3, 0].set_ylabel(f"Error [{unit}|", fontsize=16)

    if save_file_name:
        plt.savefig(f"{save_file_name}", bbox_inches='tight', pad_inches=0.1)


def visualize_quantitative_temporal(temp_tot, temp_t2m, temp_slp, scale_back=False, save_file_name=None):
    """

    :param temp_tot:
    :param temp_t2m:
    :param temp_slp:
    :param scale_back:
    :param save_file_name:
    :return:
    """
    unit = "z"
    if scale_back:
        # Scaling back for error only requires sigma, the mu gets eliminated in the subtraction
        temp_t2m = temp_t2m * np.abs(T2M_SIGMA)
        temp_slp = temp_slp * np.abs(SLP_SIGMA)

    fig = plt.figure(figsize=(16, 6))

    x_tick_pos = np.arange(0, temp_tot.shape[0], 365)
    x_ticks = DATES[x_tick_pos]

    ax1 = fig.add_subplot(311)
    ax1.plot(temp_tot, label="Total")
    ax1.grid(True)
    ax1.set_xlim(0, temp_tot.shape[0])
    ax1.set_xticks(x_tick_pos, x_ticks)
    ax1.set_ylabel(f"Total [{unit}]", fontsize=16)

    ax2 = fig.add_subplot(312)
    ax2.plot(temp_t2m, label="t2m")
    ax2.grid(True)
    ax2.set_xlim(0, temp_t2m.shape[0])
    ax2.set_xticks(x_tick_pos, x_ticks)
    ax2.set_ylabel(f"t2m [{'°C' if scale_back else unit}]", fontsize=16)

    ax3 = fig.add_subplot(313)
    ax3.plot(temp_slp, label="slp")
    ax3.grid(True)
    ax3.set_xlim(0, temp_slp.shape[0])
    ax3.set_xticks(x_tick_pos, x_ticks)
    ax3.set_ylabel(f"slp [{'Pa' if scale_back else unit}]", fontsize=16)

    if save_file_name:
        plt.savefig(f"{save_file_name}", bbox_inches='tight', pad_inches=0.1)


def visualize_quantitative_spatial(spat_total_err, spat_t2m_err, spat_slp_err, scale_back=False, save_file_name=None):
    fig = plt.figure(figsize=(16, 6))

    unit = "z"
    if scale_back:
        # Scaling back for error only requires sigma, the mu gets eliminated in the subtraction
        spat_t2m_err = spat_t2m_err * np.abs(T2M_SIGMA)
        spat_slp_err = spat_slp_err * np.abs(SLP_SIGMA)

    ax1 = fig.add_subplot(131)
    im1 = ax1.matshow(spat_total_err, cmap='Reds')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    ax1.set_title(f"Total [{unit}]", fontsize=16)

    ax2 = fig.add_subplot(132)
    im2 = ax2.matshow(spat_t2m_err, cmap='Reds')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)
    ax2.set_title(f"t2m [{'°C' if scale_back else unit}]", fontsize=16)

    ax3 = fig.add_subplot(133)
    im3 = ax3.matshow(spat_slp_err, cmap='Reds')
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3)
    ax3.set_title(f"slp [{'Pa' if scale_back else unit}]", fontsize=16)

    if save_file_name:
        plt.savefig(f"{save_file_name}", bbox_inches='tight', pad_inches=0.1)


def visualize_errors_as_function_of_elevation(elev, spat_total_err, spat_t2m_err, spat_slp_err,
                                              correlations, scale_back=False, save_file_name=None):
    fig = plt.figure(figsize=(16, 6))

    unit = "z"
    if scale_back:
        # Scaling back for error only requires sigma, the mu gets eliminated in the subtraction
        spat_t2m_err = spat_t2m_err * np.abs(T2M_SIGMA)
        spat_slp_err = spat_slp_err * np.abs(SLP_SIGMA)

    correlations = list(map(lambda x: round(x, 2), correlations))

    ax1 = fig.add_subplot(131)
    ax1.scatter(elev, spat_total_err)
    ax1.set_title(f"Total [{unit}], r={correlations[0]}", fontsize=16)
    ax1.grid(True)
    ax1.set_ylabel("Error")
    ax1.set_xlabel("Elevation")

    ax2 = fig.add_subplot(132)
    ax2.scatter(elev, spat_t2m_err)
    ax2.set_title(f"t2m [{'°C' if scale_back else unit}], r={correlations[1]}", fontsize=16)
    ax2.grid(True)
    ax2.set_ylabel("Error")
    ax2.set_xlabel("Elevation")

    ax3 = fig.add_subplot(133)
    ax3.scatter(elev, spat_slp_err)
    ax3.set_title(f"slp [{'Pa' if scale_back else unit}], r={correlations[2]}", fontsize=16)
    ax3.grid(True)
    ax3.set_ylabel("Error")
    ax3.set_xlabel("Elevation")

    if save_file_name:
        plt.savefig(f"{save_file_name}", bbox_inches='tight', pad_inches=0.1)
