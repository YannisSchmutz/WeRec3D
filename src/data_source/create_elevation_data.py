import numpy as np
import rioxarray
from skimage.measure import block_reduce


# Define reduced size window
from src.config import LAT_START, LAT_END, LON_START, LON_END

Y_LEN = 40
X_LEN = 68

SRC_FILE = "elevation/ETOPO1_Ice_g_geotiff.tif"
DST_FILE = "elevation/elevation_mat.npy"


def main():
    # Read world elevation matrix
    print(f"[*] Read source file")
    rds = rioxarray.open_rasterio(SRC_FILE)

    # Cut-off area of interest
    print(f"[*] Reduce spatial size")
    y_values = rds['y'].to_numpy()
    x_values = rds['x'].to_numpy()
    y_values = y_values[np.where((y_values <= LAT_START) & (y_values > LAT_END))]
    x_values = x_values[np.where((x_values >= LON_START) & (x_values < LON_END))]
    # Otherwise the value "33.000000000000014" is also considered and messes up the height
    y_values = y_values[:-1]

    europe_rds = rds.sel(dict(x=x_values, y=y_values))
    europe_mat = europe_rds.to_numpy()
    europe_mat = europe_mat[0]

    print(f"[*] Clip too negative values")
    europe_mat = europe_mat.clip(min=-100)

    print(f"[*] Reduce spatial granularity")
    y_pooler = int(europe_mat.shape[0] / Y_LEN)
    x_pooler = int(europe_mat.shape[1] / X_LEN)
    europe_mat = block_reduce(europe_mat,
                              block_size=(y_pooler, x_pooler),
                              func=np.mean)
    print(f"[*] Resulted shape {europe_mat.shape}")

    print("[*] Perform min-max scaling")
    europe_mat = (europe_mat - europe_mat.min()) / (europe_mat.max() - europe_mat.min())

    # Save
    print(f"[*] Save file...")
    np.save(DST_FILE, europe_mat)


if __name__ == "__main__":
    main()
