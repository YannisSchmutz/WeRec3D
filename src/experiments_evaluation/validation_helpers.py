import numpy as np

T2M_MU = 7.9444466
T2M_SIGMA = 8.895299
SLP_MU = 101334.484
SLP_SIGMA = 1098.013


def scale_t2m_back(var, for_error=False):
    if for_error:
        # Scaling back for error only requires sigma, the mu gets eliminated in the subtraction
        return np.multiply(var, T2M_SIGMA)
    return np.add(np.multiply(var, T2M_SIGMA),  T2M_MU)


def scale_slp_back(var, for_error=False):
    if for_error:
        # Scaling back for error only requires sigma, the mu gets eliminated in the subtraction
        return np.multiply(var, SLP_SIGMA)
    return np.add(np.multiply(var, SLP_SIGMA),  SLP_MU)


def prepare_quantitative_samples1(mat, f=5, seq_reshape=True):
    """

    Expected shape: (Time, Hight, Width, Channels)
    """
    # Crop to base window
    # TODO: Base window cropping values as GLOBALS?
    mat = mat[:, 6:38, 2:66]
    base_shape = mat.shape

    nbr_samples = base_shape[0] // f
    cut_off = nbr_samples * f
    # Make sure it is dividable through "f"
    mat = mat[:cut_off]
    # Reshape to samples, frames, ...
    if seq_reshape:
        mat = np.reshape(mat, (nbr_samples, f, base_shape[1], base_shape[2], base_shape[3]))
    return mat


def calc_total_errors(y, pred):
    abs_err = np.abs(np.subtract(pred, y))

    total_err = np.mean(abs_err)
    total_variable_error = np.mean(abs_err, axis=tuple([shp for shp in range(len(abs_err.shape) - 1)]))
    return total_err, total_variable_error[0], total_variable_error[1]


def calc_spatial_errors(y, pred):
    abs_err = np.abs(np.subtract(pred, y))
    spatial_err = np.mean(abs_err, axis=(0, -1))
    spatial_variable_error = np.mean(abs_err, axis=0)
    return spatial_err, spatial_variable_error[..., 0], spatial_variable_error[..., 1]


def calc_temporal_errors(y, pred):
    abs_err = np.abs(np.subtract(pred, y))
    temporal_error = np.mean(abs_err, axis=tuple([shp for shp in range(1, len(abs_err.shape))]))
    temporal_variable_error = np.mean(abs_err, axis=tuple([shp for shp in range(1, len(abs_err.shape) - 1)]))
    return temporal_error, temporal_variable_error[..., 0], temporal_variable_error[..., 1]
