from ofcsst.utils.ids import BINARY_2VN

# Simulation environment parameters
TASK_ID = BINARY_2VN
REWARD = 1.0  # Reward value in case of Hit
PUNISHMENT = -1.0  # Reward value in case of FA
NUMBER_SIGNAL = 2  # Standard number of signal neurons
NUMBER_DISTRACTOR = 128  # Standard number of distractor neurons
NR_TRIALS: int = 600  # Number of trials simulated for each rule
NR_SEEDS = 64  # Number of seeds for hyperparameter scans, computational benchmarks and replicating neuronal variability
NR_SIM_MICE = 10  # Number of seeds to replicate mouse number variability

# Processing parameters
SIGNIFICANCE_THRESHOLD = (
    0.05  # Adjusted p-value threshold to mark an effect as statistically significant
)
PERF_MAV_N: int = (
    100  # Number of trials to include in moving average to compute performance
)
EXPERT_PERFORMANCE = 85  # Performance threshold to reach expertise for model
EXPERT_D: float = 1.5  # Discriminability threshold to reach expertise for mice

# Model parameters
POLICY_WEIGHT = 10.0  # Synaptic strength cap for policy weights
TEMPERATURE = 10.0  # Temperature of softmax
MAX_GAIN = 10.0  # Multiplicative gain cap
MAX_SD = 0.5  # Maximal variance for the binary discrimination task
SD_LR = 0.03  # Learning rate of uncertainty estimator
OFC_LR = 0.03  # Learning rate of the OFC (i.e. of the CPE to predict the CL)
APICAL_LR = 0.1  # Learning rate of the apical afferent (\eta^\mathrm{ap})
SST0 = 0.1  # Baseline SST interneuron activity
CPE_SWITCH = 1.0  # Threshold activity for the lOFC activity (i.e. the CPE) to trigger a context switch
CONFIDENCE_SWITCH = (
    0.95  # Confidence threshold to allow the lOFC to trigger a confidence switch
)
