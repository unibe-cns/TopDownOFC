from typing import Literal

# Keys for environment variables
SNKey = Literal[
    "n_distractor",
    "n_signal",
    "distractor_amplitude",
    "signal_amplitude",
    "signal_to_noise_ratio",
]
NR_DISTRACTOR: SNKey = "n_distractor"
NR_SIGNAL: SNKey = "n_signal"
DISTRACTOR_GAIN: SNKey = "distractor_amplitude"
SIGNAL_GAIN: SNKey = "signal_amplitude"
SNR: SNKey = "signal_to_noise_ratio"

# Keys for learning rates
TrainKey = Literal["lr_pg", "lr_v", "lr_q", "lr_ap", "lr_sd", "lr_ofc"]
LEARNING_RATE_PG: TrainKey = "lr_pg"
LEARNING_RATE_V: TrainKey = "lr_v"
LEARNING_RATE_Q: TrainKey = "lr_q"
LEARNING_RATE_APICAL: TrainKey = "lr_ap"
LEARNING_RATE_SD: TrainKey = "lr_sd"
LEARNING_RATE_OFC: TrainKey = "lr_ofc"
LEARNING_RATE_KEYS: list[TrainKey] = [
    LEARNING_RATE_PG,
    LEARNING_RATE_V,
    LEARNING_RATE_Q,
    LEARNING_RATE_APICAL,
    LEARNING_RATE_SD,
    LEARNING_RATE_OFC,
]

# Simulation keys
SimulationKey = Literal["seed"]
SEED: SimulationKey = "seed"

# Keys for simulation outcomes that can be selected for during parameter scanning
OutcomeKey = Literal["loss", "performance", "expert_fraction", "expert_time"]
LOSS: OutcomeKey = "loss"
PERFORMANCE: OutcomeKey = "performance"
EXPERT_FRAC: OutcomeKey = "expert_fraction"
EXPERT_TIME: OutcomeKey = "expert_time"

# Keys for parameters determining the effect of lOFC activity onto SST interneurons
OFCParamKey = Literal[
    "ofc_amplification_constant", "threshold_activity", "ofc_in_multi_constant"
]
SST_B: OFCParamKey = "threshold_activity"
SST_W: OFCParamKey = "ofc_amplification_constant"
OFC_K_IN: OFCParamKey = "ofc_in_multi_constant"

# Keys for parameters only used for control algorithms
ControlAlgKey = Literal["rw_k"]
RW_K: ControlAlgKey = "rw_k"

# Keys of variables that are tracked by the logger and will be used to identify tables and columns in the database
LogKey = Literal["gain", "policy_weight", "ofc_activity"]
AGENT_GAIN: LogKey = "gain"
AGENT_POLICY_WEIGHT: LogKey = "policy_weight"
OFC_ACTIVITY: LogKey = "ofc_activity"

# Keys of individual figures
FigureKey = Literal["3", "s7", "s8"]
FIGURE_3: FigureKey = "3"
FIGURE_S7: FigureKey = "s7"
FIGURE_S8: FigureKey = "s8"

Key = SNKey | TrainKey | SimulationKey | OutcomeKey | OFCParamKey | ControlAlgKey
