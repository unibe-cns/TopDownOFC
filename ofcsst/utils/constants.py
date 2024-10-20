from ofcsst.utils.ids import BINARY_2VN

# Simulation environment parameters
TASK_ID = BINARY_2VN
REWARD = 1.
PUNISHMENT = -1.
NUMBER_SIGNAL = 2
NUMBER_DISTRACTOR = 128
NR_TRIALS: int = 600
NR_SEEDS = 64

# Processing parameters
SIGNIFICANCE_THRESHOLD = 0.05
PERF_MAV_N: int = 100
EXPERT_PERFORMANCE = 85
EXPERT_D = 1.5

# Model parameters
POLICY_WEIGHT_CLAMP = 10.
Q_TEMPERATURE = 10.
MAX_GAIN = 10.
MAX_SD = 0.5
SD_LR = 0.03
OFC_LR = 0.03
APICAL_LR = 0.1
SST0 = 0.1
CPE_SWITCH = 1.
CONFIDENCE_SWITCH = 0.95
