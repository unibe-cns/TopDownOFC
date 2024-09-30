import numpy as np
from ofcsst.utils import constants


def get_performance(outcomes: np.ndarray) -> np.ndarray:
    """Compute performance trace as a moving average of the rewarded outcomes"""
    conv_filter = 100 * np.ones(constants.PERF_MAV_N) / constants.PERF_MAV_N
    padded_outcomes = np.concatenate((outcomes[:int(constants.PERF_MAV_N / 2)], outcomes,
                                      outcomes[-int(constants.PERF_MAV_N / 2) + 1:]))
    return np.convolve(a=padded_outcomes, v=conv_filter, mode='valid')


def get_expert_t(performances: np.ndarray):
    """Get ID of the first trial where the expert performance was reached"""
    expert_ts = performances > constants.EXPERT_PERFORMANCE
    if np.sum(expert_ts) > 0:
        return np.argmax(expert_ts)
    else:
        return np.nan
