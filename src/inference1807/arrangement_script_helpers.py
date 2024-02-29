
import numpy as np

import os, sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath("__init__.py"))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()
sys.path.append(os.path.dirname(SRC_DIR))

from src.config import BASE_LAT_START, BASE_LAT_END, BASE_LON_START
from src.config import H as BASE_HEIGHT


def add_matrix_ids_to_coords(coords):
    coords['x'] = coords['lon'].pipe(np.floor) + abs(BASE_LON_START)
    coords['y'] = coords['lat'].apply(lambda x: ((x - BASE_LAT_START) / (BASE_LAT_END - BASE_LAT_START)) * BASE_HEIGHT).pipe(np.floor)
    coords = coords.astype({'x': 'int', 'y': 'int'})
    coords = coords.set_index('id')
    return coords
