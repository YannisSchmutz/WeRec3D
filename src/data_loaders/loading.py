import numpy as np
import os
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()

DATA_DIR_A = str(SRC_DIR) + "/data_sets/variant_a/"
DATA_DIR_B = str(SRC_DIR) + "/data_sets/variant_b/"

TRAIN_SET_NAME = "train.npy"
VAL_SET_NAME = "validation.npy"
TEST_SET_NAME = "test.npy"

MASK_DIR = str(SRC_DIR) + "/masks/"
ELEVATION_FILE_PATH = str(SRC_DIR) + "/data_source/elevation/elevation_mat.npy"

# Use the same values for replacing train and val masked cells.
T2M_REPLACE_VALUE_A = -7.7686565e-08   # train t2m mean of z-scaled data variant A
MSL_REPLACE_VALUE_A = -1.3009013e-08  # train msl mean of z-scaled data variant A
VARIABLES_REPLACE_VALUE_A = np.array([T2M_REPLACE_VALUE_A, MSL_REPLACE_VALUE_A])

# Use the same values for replacing train and val masked cells.
T2M_REPLACE_VALUE_B = 0.000001345   # train t2m mean of z-scaled data variant B
MSL_REPLACE_VALUE_B = -0.000066427  # train msl mean of z-scaled data variant B
VARIABLES_REPLACE_VALUE_B = np.array([T2M_REPLACE_VALUE_B, MSL_REPLACE_VALUE_B])

PI_INIT_VAL_DIR = str(SRC_DIR) + "/data_loaders/pi_init/"


def get_elevation_data(base_window_only=False):
    """
    :param base_window_only: If true, only return the fixed position window.
    :return:
    """
    elev_mat = np.load(ELEVATION_FILE_PATH)
    if base_window_only:
        elev_mat = elev_mat[6:38, 2:66]
    return elev_mat.astype("float32")


def _create_y_set(variables_set, mask_set):
    return np.concatenate((variables_set, mask_set), axis=-1).astype('float32')


def _create_x_set(variables_set, mask_set, variant, pi_replacement):
    """
    Creates X set with imputed masked variables and mask in last channels.
    BEWARE: Per default numpy arrays are passed by reference!
        Thus pass a copy of the variable_set if you need it further-

    :param variables_set: Variables.
    :param mask_set: Masks.
    :return:
    """
    if variant == "a":
        variables_replace_value = VARIABLES_REPLACE_VALUE_A
        if pi_replacement:
            variables_replace_value = np.load(f"{PI_INIT_VAL_DIR}averages_a.npy")
    elif variant == "b":
        variables_replace_value = VARIABLES_REPLACE_VALUE_B
        if pi_replacement:
            variables_replace_value = np.load(f"{PI_INIT_VAL_DIR}averages_b.npy")
    else:
        raise ValueError(f"Variant {variant} not supported")

    # Create masked set
    masked_mat = np.ma.masked_array(variables_set, mask=mask_set)
    # Replace masked values with mean value of corresponding variable
    masked_mat = masked_mat.astype(np.float64).filled(variables_replace_value)
    replaced_values = np.ma.getdata(masked_mat)
    return np.concatenate((replaced_values, mask_set), axis=-1).astype('float32')


def get_train_val_sets(variant, percentage, missing_type="mcar", pi_replacement=False):
    if variant == "a":
        base_dir = DATA_DIR_A
    elif variant == "b":
        base_dir = DATA_DIR_B
    else:
        raise NotImplemented("Choose a or b!")

    if missing_type == "mcar":
        mask_dir = MASK_DIR + "mcar/"
    else:
        mask_dir = MASK_DIR + "mnar/"

    # --- TRAIN ---
    print(f"Loading train variables file: {TRAIN_SET_NAME}")
    train_variables = np.load(base_dir + TRAIN_SET_NAME)
    print(f"Loading train mask file: train_mask_{percentage}p.npy")
    train_masks = np.load(mask_dir + f"train_mask_{percentage}p.npy")
    print("Creating x and y training sets...")
    x_train = _create_x_set(np.copy(train_variables), train_masks, variant, pi_replacement)
    y_train = _create_y_set(train_variables, train_masks)

    # --- VAL ---
    print(f"Loading validation variables file: {VAL_SET_NAME}")
    val_variables = np.load(base_dir + VAL_SET_NAME)
    print(f"Loading validation mask file: validation_mask_{percentage}p.npy")
    val_masks = np.load(mask_dir + f"validation_mask_{percentage}p.npy")
    print("Creating x and y validation sets...")
    x_validation = _create_x_set(np.copy(val_variables), val_masks, variant, pi_replacement)
    y_validation = _create_y_set(val_variables, val_masks)

    return {'train': {'x': x_train,
                      'y': y_train
                      },
            'validation': {'x': x_validation,
                           'y': y_validation}
            }


def get_val_sets(variant, percentage, missing_type="mcar", include_elevation=False, pi_replacement=False):
    if variant == "a":
        base_dir = DATA_DIR_A
    elif variant == "b":
        base_dir = DATA_DIR_B
    else:
        raise NotImplemented("Choose a or b!")

    if missing_type == "mcar":
        mask_dir = MASK_DIR + "mcar/"
    else:
        mask_dir = MASK_DIR + "mnar/"

    # --- VAL ---
    print(f"Loading validation variables file: {VAL_SET_NAME}")
    val_variables = np.load(base_dir + VAL_SET_NAME)
    print(f"Loading validation mask file: validation_mask_{percentage}p.npy")
    val_masks = np.load(mask_dir + f"validation_mask_{percentage}p.npy")
    print("Creating x and y validation sets...")
    x_validation = _create_x_set(np.copy(val_variables), val_masks, variant, pi_replacement)
    y_validation = _create_y_set(val_variables, val_masks)

    if include_elevation:
        elev = get_elevation_data()
        # Create sequence and channel dimension
        elev = np.expand_dims(elev, axis=(0, -1))
        elev = np.repeat(elev, repeats=x_validation.shape[0], axis=0)
        x_validation = np.concatenate((x_validation, elev), axis=-1)

    return x_validation, y_validation


def get_qualitative_sets(variant="b", percentage="50", missing_type="mcar", include_elevation=False,
                         pi_replacement=False):
    # Static here in function for now... refactor if needed!
    quali_indices = [229, 230, 231, 232, 233]
    quali_dates = ['1955-08-18', '1955-08-19', '1955-08-20', '1955-08-21', '1955-08-22']

    x_val, y_val = get_val_sets(variant, percentage, missing_type,
                                include_elevation=include_elevation, pi_replacement=pi_replacement)
    x_val = x_val[quali_indices, 6:38, 2:66]
    y_val = y_val[quali_indices, 6:38, 2:66]

    # For now handle the cropping of the space here manually... Refactor if needed!
    return {'x': x_val,
            'y': y_val,
            'dates': quali_dates}


def get_test_sets(percentage="99"):
    mask_dir = MASK_DIR + "mnar/"

    print(f"Loading test variables file: {TEST_SET_NAME}")
    test_variables = np.load(DATA_DIR_B + TEST_SET_NAME)
    print(f"Loading test mask file: test_mask_{percentage}p.npy")
    test_masks = np.load(mask_dir + f"test_mask_{percentage}p.npy")
    print("Creating x and y validation sets...")
    x_test = _create_x_set(np.copy(test_variables), test_masks, "b", False)
    y_test = _create_y_set(test_variables, test_masks)

    elev = get_elevation_data()
    # Create sequence and channel dimension
    elev = np.expand_dims(elev, axis=(0, -1))
    elev = np.repeat(elev, repeats=x_test.shape[0], axis=0)
    x_test = np.concatenate((x_test, elev), axis=-1)

    return x_test, y_test