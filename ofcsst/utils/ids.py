from typing import Literal

# Simulation type identifying the model details such as the gain modulation apparatus or the action selection policy
SimulationType = Literal[
    "pg_no_gain",
    "pg_no_gain_deep",
    "pg_gain",
    "pure_pg_gain",
    "pure_q_gain",
    "pg_rw",
    "soft_q_no_gain",
    "soft_q_gain",
    "convex_no_gain",
    "convex_gain",
    "convex_gain_reversal_reset",
    "convex_actor_fake_ofc",
    "convex_ofc",
    "ofc_xap_switch",
    "convex_ofc_no_vip",
]
SIM_PG_NO_GAIN: SimulationType = "pg_no_gain"  # Policy Gradient without gain modulation
SIM_PG_NO_GAIN_DEEP: SimulationType = (
    "pg_no_gain_deep"  # Bottom-up plasticity without gain modulation
)
SIM_PG_GAIN: SimulationType = (
    "pg_gain"  # Policy Gradient with our gain modulation (without OFC)
)
SIM_PURE_PG_GAIN: SimulationType = (
    "pure_pg_gain"  # Policy Gradient with gain modulation from backpropagated PG loss
)
SIM_PG_RW: SimulationType = "pg_rw"  # Policy Gradient with Rescorla-Wagner gain
SIM_SQ_NO_GAIN: SimulationType = "soft_q_no_gain"  # Q-actor without gain modulation
SIM_SQ_GAIN: SimulationType = (
    "soft_q_gain"  # Q-actor with our gain modulation (without OFC)
)
SIM_PURE_SQ_GAIN: SimulationType = (
    "pure_q_gain"  # Q-actor with backprop gain from Q loss
)
SIM_CONVEX_NO_GAIN: SimulationType = "convex_no_gain"
SIM_CONVEX_GAIN: SimulationType = "convex_gain"
SIM_CONVEX_GAIN_REV_RESET: SimulationType = "convex_gain_reversal_reset"
SIM_CONVEX_FAKE_OFC: SimulationType = "convex_actor_fake_ofc"
SIM_CONVEX_OFC: SimulationType = "convex_ofc"
SIM_CONVEX_OFC_VIP_KO: SimulationType = "convex_ofc_no_vip"
SIM_OFC_XAP_SWITCH: SimulationType = "ofc_xap_switch"
OFC_SST_SIMULATIONS: list[SimulationType] = [
    SIM_CONVEX_OFC,
    SIM_CONVEX_OFC_VIP_KO,
]
FAKE_OFC_SIMULATIONS: list[SimulationType] = [
    SIM_CONVEX_GAIN_REV_RESET,
    SIM_CONVEX_FAKE_OFC,
]
TD_LR_SIMULATIONS: list[SimulationType] = [
    SIM_PG_GAIN,
    SIM_CONVEX_GAIN,
    SIM_CONVEX_GAIN_REV_RESET,
    SIM_CONVEX_FAKE_OFC,
    SIM_OFC_XAP_SWITCH,
    SIM_CONVEX_OFC,
    SIM_CONVEX_OFC_VIP_KO,
    SIM_SQ_GAIN,
    SIM_PG_RW,
]

# Task identifier
TaskID = Literal["Binary_2vsN"]
BINARY_2VN: TaskID = "Binary_2vsN"

# Identifier for the parameter scanning paradigm
ScanType = Literal[
    "vary_distraction_2d",
    "vary_distraction_1d",
    "final_conditions",
    "no_distractor",
    "convex_final",
    "vary_all",
    "vary_top_down",
]
ST_VARY_DISTRACTION_2D: ScanType = "vary_distraction_2d"
ST_VARY_DISTRACTOR_1D: ScanType = "vary_distraction_1d"
ST_FINAL: ScanType = "final_conditions"
ST_NO_DISTRACTOR: ScanType = "no_distractor"
ST_TD: ScanType = "vary_top_down"
ST_ALL: ScanType = "vary_all"

# Identifier for the optimizer
OptimizerType = Literal["sgd", "adam", "adagrad", "rmsprop", "adamax", "adadelta"]
OP_SGD: OptimizerType = "sgd"
OP_ADAM: OptimizerType = "adam"
OP_ADAGRAD: OptimizerType = "adagrad"
OP_RMSP: OptimizerType = "rmsprop"
OP_ADAMAX: OptimizerType = "adamax"
OP_ADADELTA: OptimizerType = "adadelta"
OPTIMIZERS: list[OptimizerType] = [
    OP_SGD,
    OP_ADAM,
    OP_ADAGRAD,
    OP_RMSP,
    OP_ADAMAX,
    OP_ADADELTA,
]

# Identifier for the variable logging, which determines what variables are tracked during the simulation
LogType = Literal["ofc", "agent", "gain", "convex", "all", "rel_gain"]
LT_OFC: LogType = "ofc"
LT_AGENT: LogType = "agent"
LT_GAIN: LogType = "gain"
LT_RGAIN: LogType = "rel_gain"
LT_CONVEX: LogType = "convex"
LT_ALL: LogType = "all"

# Identifiers for types of mice according to their genetic modifications
MouseType = Literal["WT", "OFC_KO", "OFC_WT", "VIP_KO", "VIP_WT"]
MT_WT: MouseType = "WT"
MT_OFC_KO: MouseType = "OFC_KO"
MT_OFC_WT: MouseType = "OFC_WT"
MT_VIP_KO: MouseType = "VIP_KO"
MT_VIP_WT: MouseType = "VIP_WT"

# Identifiers for learning rules and how many reversals have taken place
LearningRule = Literal["learning", "reversal", "reversal2"]
LR_L: LearningRule = "learning"
LR_R: LearningRule = "reversal"
LR_R2: LearningRule = "reversal2"


def get_optimizer(optimizer_string: str) -> OptimizerType:
    """Match a string from a scan job file to its OptimizerType identifier.

    :param optimizer_string: optimizer name as written in the job file.
    :returns: the matching OptimizerType.
    :raises ValueError: if the string names no known optimizer.
    """
    for optimizer in OPTIMIZERS:
        if optimizer_string == optimizer:
            return optimizer
    raise ValueError(
        f"'{optimizer_string}' is not a recognized optimizer out of {OPTIMIZERS}"
    )
