import numpy as np
import click
import pandas as pd
from itertools import product
from math import sqrt
import os, sys
from pathlib import Path
from copy import deepcopy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()
sys.path.append(os.path.dirname(SRC_DIR))

from src.config import LAT_START, LAT_END, LON_START

STATIONS_1807 = str(SRC_DIR) + ("/metadata/station_1807_matrix_indi"
                                "ces.csv")

TARGET_DIR = "../masks/"
TVT_DAYS = [20_454, 3_653, 1_826]
HEIGHT = 40
WIDTH = 68
CHANNELS = 2


def create_mask(missing_ratio, nbr_days, height, width, nbr_predictors):
    p_mask = np.random.choice(np.array([0, 1]),
                              size=(nbr_days, height, width, nbr_predictors),
                              replace=True,
                              p=[1 - missing_ratio, missing_ratio])
    return p_mask.astype('byte')


def create_mcar_basic_masks():
    for i, split_name in enumerate(['train', 'validation', 'test']):
        print(f"Create masks for {split_name}-set")
        for p in list(range(10, 100, 10)) + [99]:
            print(f"    {p}%")
            nbr_days = TVT_DAYS[i]

            missing_rate = p / 100

            mask = create_mask(missing_ratio=missing_rate,
                               nbr_days=nbr_days,
                               height=HEIGHT,
                               width=WIDTH,
                               nbr_predictors=CHANNELS)

            file_name = f"{split_name}_mask_{p}p.npy"
            print(f"    Saving {file_name}")

            save_path = f"{TARGET_DIR}mcar/" + file_name
            np.save(save_path, mask)


def create_single_mcar_masks(percentage):
    for i, split_name in enumerate(['train', 'validation', 'test']):
        print(f"Create {percentage}% MCAR mask for {split_name}-set")
        nbr_days = TVT_DAYS[i]
        missing_rate = percentage / 100
        mask = create_mask(missing_ratio=missing_rate,
                           nbr_days=nbr_days,
                           height=HEIGHT,
                           width=WIDTH,
                           nbr_predictors=CHANNELS)
        file_name = f"{split_name}_mask_{percentage}p.npy"
        print(f"    Saving {file_name}")

        save_path = f"{TARGET_DIR}mcar/" + file_name
        np.save(save_path, mask)


# quite ugly code, but works...
def create_mnar_masks():
    def get_additional_coordinates(init_coords):
        addon_coord_pairs = list(map(lambda pair: list(pair), init_coords))
        percentage_cord_mapping = {}
        radius = 1
        observation_counter = len(init_coords)
        # Initial coords --> 99% missing. Thus leave id for 99% out
        for reversed_percent_id in range(len(observed_counts)-2, -1, -1):
            current_mask_created = False
            while True:
                possible_adjuster = list(product(range(-radius, radius+1), repeat=2))
                for y_coord, x_coord in init_coords:
                    for y_adj, x_adj in possible_adjuster:
                        pyc = y_coord + y_adj
                        pxc = x_coord + x_adj
                        if (0 <= pyc < HEIGHT) and (0 <= pxc < WIDTH):
                            if sqrt(y_adj**2 + x_adj**2) <= radius:
                                if [pyc, pxc] not in addon_coord_pairs:
                                    observation_counter += 1
                                    addon_coord_pairs.append([pyc, pxc])
                                    if observation_counter == observed_counts[reversed_percent_id]:
                                        print(f"Reached mask {percentages_missing[reversed_percent_id]}% missing at {observation_counter} observations")
                                        percentage_cord_mapping[percentages_missing[reversed_percent_id]] = deepcopy(addon_coord_pairs)
                                        current_mask_created = True
                                        break
                    if current_mask_created:
                        break
                else:
                    radius += 1
                if current_mask_created:
                    break
        return percentage_cord_mapping

    # Read positions of station data and transform them to the corresponding cells in the mask
    coords = pd.read_csv(STATIONS_1807)
    coords = coords[['lat', 'lon', 'variable']]
    coords['lon'] = coords['lon'].pipe(np.floor) + abs(LON_START)
    coords['lat'] = coords['lat'].apply(lambda x: ((x - LAT_START) / (LAT_END - LAT_START)) * HEIGHT).pipe(np.floor)
    t2m_init_coords = coords[coords['variable'] == 'ta'][['lat', 'lon']].to_numpy().astype('int')
    msl_init_coords = coords[coords['variable'] == 'slp'][['lat', 'lon']].to_numpy().astype('int')

    percentages_missing = list(range(10, 100, 10)) + list(range(92, 100, 2)) + [99]
    N = HEIGHT * WIDTH
    percentages_observed = list(map(lambda p: 100-p, percentages_missing))
    observed_counts = list(map(lambda c: int(c*N/100), percentages_observed))

    t2m_percentage_coords = get_additional_coordinates(t2m_init_coords)
    msl_percentage_coords = get_additional_coordinates(msl_init_coords)

    t2m_percentage_coords[99] = t2m_init_coords
    msl_percentage_coords[99] = msl_init_coords

    for missing_percentage in percentages_missing:
        t2m_mat = np.ones([HEIGHT, WIDTH])
        msl_mat = np.ones([HEIGHT, WIDTH])

        t2m_coords = np.array(t2m_percentage_coords[missing_percentage])
        msl_coords = np.array(msl_percentage_coords[missing_percentage])
        t2m_mat[t2m_coords[:, 0], t2m_coords[:, 1]] = 0
        msl_mat[msl_coords[:, 0], msl_coords[:, 1]] = 0
        mat = np.concatenate((
            np.expand_dims(t2m_mat, axis=-1),
            np.expand_dims(msl_mat, axis=-1)), axis=-1)
        for i, split_name in enumerate(['train', 'validation', 'test']):
            print(f"Create masks for {split_name}-set, {missing_percentage}%")
            nbr_days = TVT_DAYS[i]
            mask = np.repeat(np.expand_dims(mat, axis=0), nbr_days, axis=0)

            file_name = f"{split_name}_mask_{missing_percentage}p.npy"
            print(f"    Saving {file_name}")

            save_path = f"{TARGET_DIR}mnar/" + file_name
            np.save(save_path, mask)


@click.command()
@click.option('-m', '--missing_mechanism', required=True, type=str)
def main(missing_mechanism):
    if missing_mechanism == "mcar":
        create_mcar_basic_masks()
    elif missing_mechanism == "mnar":
        create_mnar_masks()
    else:
        print("Only mcar or mnar supported!")


if __name__ == '__main__':
    main()
