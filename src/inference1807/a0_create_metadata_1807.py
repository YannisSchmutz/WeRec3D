"""
Arrangement Script 0
====================
Creates Station-Matrix Indices (lat, lon, y, x) mapping.
"""

import os, sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath("__init__.py"))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()
sys.path.append(os.path.dirname(SRC_DIR))

import pandas as pd
from arrangement_script_helpers import add_matrix_ids_to_coords
from src.config import METADATA_DIR


def main():
    # Read stations and coordinates
    diff_ta = pd.read_csv("bias_data/diff_station_raster_ta.txt", delimiter=" ")
    diff_slp = pd.read_csv("bias_data/diff_station_raster_slp.txt", delimiter=" ")

    # Store station coordinates
    ta_coords = add_matrix_ids_to_coords(diff_ta[['id', 'lat', 'lon']])
    slp_coords = add_matrix_ids_to_coords(diff_slp[['id', 'lat', 'lon']])

    # Cleanup
    ta_coords['variable'] = "ta"
    ta_coords.index = ta_coords.index.str.replace('_ta', '')
    slp_coords['variable'] = "slp"
    slp_coords.index = slp_coords.index.str.replace('_slp', '')

    station_coordinate_df = pd.concat([ta_coords, slp_coords])
    station_coordinate_df = station_coordinate_df.sort_index()
    station_coordinate_df.to_csv(str(SRC_DIR) + METADATA_DIR + "station_1807_matrix_indices.csv")


if __name__ == "__main__":
    main()
