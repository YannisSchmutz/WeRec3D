METADATA_DIR = "/metadata/"

# STATION_INDICES = "metadata/station_{}_matrix_indices.csv"
ORIG_TEST_SET = "/data_sets/variant_b/test.npy"

VARIABLES_SET_INFERENCE = "/inference1807/data_sets/variables_set.npy"
MASK_SET_INFERENCE = "/inference1807/masks/mask_set.npy"
VARIABLES_SET_INFERENCE_ARM = "/inference1807/data_sets/variables_set_analog_3p.npy"
MASK_SET_INFERENCE_ARM = "/inference1807/masks/mask_set_analog_3p.npy"
VARIABLES_SET_INFERENCE_ARM_WT = "/inference1807/data_sets/variables_set_analog_3p_WT.npy"
MASK_SET_INFERENCE_ARM_WT = "/inference1807/masks/mask_set_analog_3p_WT.npy"

MODEL_PATH_PLAIN = "/experiments_evaluation/ex7.2_emc_mnar_training/model_checkpoint/p99/"
MODEL_PATH_ARM_ENHANCED = "/experiments_evaluation/ex8_emc_nineties/model_checkpoint/p96/"

# Define ERA5 download window
DL_LAT_START = 75
DL_LAT_END = 30
DL_LON_START = -25
DL_LON_END = 45

# Defines 40x68 window
LAT_START = 73
LAT_END = 33
LON_START = -24
LON_END = 44

# Define reduced size window
BASE_LAT_START = 67
BASE_LAT_END = 36
BASE_LON_START = -22
BASE_LON_END = 41

# Scale back constants
TA_MU = 7.9444466
TA_SIGMA = 8.895299
SLP_MU = 101334.484
SLP_SIGMA = 1098.013

# Masked replace values
T2M_REPLACE_VALUE_B = 0.000001345
SLP_REPLACE_VALUE_B = -0.000066427

# Test Data Cropping
# DAYS_IN_1807 = 365
BASE_WINDOW_Y_CROP_1 = 6
BASE_WINDOW_Y_CROP_2 = 38
BASE_WINDOW_X_CROP_1 = 2
BASE_WINDOW_X_CROP_2 = 66


# Model parameters
F = 5  # Days per sample
H = 32
W = 64
CH = 5  # t2m, msl, msk1, msk2, elev
BS = 2  # minimum of 2 due to two GPUs


# Analogue Resampling Method
WINDOW_SIDE_SIZE = 15
DATA_DIR_B = "/data_sets/variant_b/"
TRAIN_SET_NAME = "train.npy"
WT_PROBABILITY_THRESHOLD = 0.9

