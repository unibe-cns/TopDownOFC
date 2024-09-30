from typing import Literal

# Simulation type identifying the model details such as the gain modulation apparatus or the action selection policy
SimulationType = Literal[
    'pg_no_gain', 'pg_gain', 'convex_no_gain', 'convex_gain', 'convex_gain_reversal_reset',
    'convex_actor_fake_ofc', 'convex_ofc', 'fake_td_switch', 'fake_xap_switch', 'ofc_xap_switch', 'convex_ofc_no_vip'
]
SIM_PG_NO_GAIN: SimulationType = 'pg_no_gain'
SIM_PG_GAIN: SimulationType = 'pg_gain'
SIM_CONVEX_NO_GAIN: SimulationType = 'convex_no_gain'
SIM_CONVEX_GAIN: SimulationType = 'convex_gain'
SIM_CONVEX_GAIN_REV_RESET: SimulationType = 'convex_gain_reversal_reset'
SIM_CONVEX_FAKE_OFC: SimulationType = 'convex_actor_fake_ofc'
SIM_CONVEX_OFC: SimulationType = 'convex_ofc'
SIM_CONVEX_OFC_VIP_KO: SimulationType = 'convex_ofc_no_vip'
SIM_FAKE_TD_SWITCH: SimulationType = 'fake_td_switch'
SIM_FAKE_XAP_SWITCH: SimulationType = 'fake_xap_switch'
SIM_OFC_XAP_SWITCH: SimulationType = 'ofc_xap_switch'
OFC_SST_SIMULATIONS: list[SimulationType] = [SIM_CONVEX_OFC, SIM_FAKE_TD_SWITCH, SIM_CONVEX_OFC_VIP_KO]
FAKE_OFC_SIMULATIONS: list[SimulationType] = [SIM_CONVEX_GAIN_REV_RESET, SIM_CONVEX_FAKE_OFC]
TD_LR_SIMULATIONS: list[SimulationType] = [
    SIM_PG_GAIN, SIM_CONVEX_GAIN, SIM_CONVEX_GAIN_REV_RESET, SIM_CONVEX_FAKE_OFC, SIM_OFC_XAP_SWITCH,
    SIM_CONVEX_OFC, SIM_FAKE_TD_SWITCH, SIM_CONVEX_OFC_VIP_KO
]

# Task identifier
TaskID = Literal["Binary_2vsN"]
BINARY_2VN: TaskID = "Binary_2vsN"
TASKS = [BINARY_2VN]

# Identifier for the parameter scanning paradigm
ScanType = Literal[
    'vary_distraction_2d', 'vary_distraction_1d', 'final_conditions',
    'no_distractor', 'convex_final', 'vary_all', 'vary_top_down'
]
ST_VARY_DISTRACTION_2D: ScanType = 'vary_distraction_2d'
ST_VARY_DISTRACTOR_1D: ScanType = 'vary_distraction_1d'
ST_FINAL: ScanType = 'final_conditions'
ST_NO_DISTRACTOR: ScanType = 'no_distractor'
ST_TD: ScanType = "vary_top_down"
ST_ALL: ScanType = "vary_all"

# Identifier for the optimizer
OptimizerType = Literal[
    'sgd', 'adam', 'adagrad', 'rmsprop', 'adamax', 'adadelta'
]
OP_SGD: OptimizerType = 'sgd'
OP_ADAM: OptimizerType = 'adam'
OP_ADAGRAD: OptimizerType = 'adagrad'
OP_RMSP: OptimizerType = 'rmsprop'
OP_ADAMAX: OptimizerType = 'adamax'
OP_ADADELTA: OptimizerType = "adadelta"
OPTIMIZERS: list[OptimizerType] = [OP_SGD, OP_ADAM, OP_ADAGRAD, OP_RMSP, OP_ADAMAX, OP_ADADELTA]

# Identifier for the variable logging, which determines what variables are tracked during the simulation
LogType = Literal['ofc', 'agent', 'gain', 'convex', 'all', 'rel_gain', 'selectivity']
LT_OFC: LogType = 'ofc'
LT_AGENT: LogType = 'agent'
LT_GAIN: LogType = 'gain'
LT_RGAIN: LogType = 'rel_gain'
LT_CONVEX: LogType = 'convex'
LT_SEL: LogType = 'selectivity'
LT_ALL: LogType = 'all'


def get_literal(literal_string: str):
    """
    Take a string and match it to the corresponding literal string identifier

    Parameters
    ----------
    literal_string: string to use as literal

    Returns a literal string
    -------

    """
    literals = OPTIMIZERS
    for literal in literals:
        if literal_string == literal:
            return literal
    raise ValueError(f"'{literal_string}' is not a recognized literal out of {literals}")
