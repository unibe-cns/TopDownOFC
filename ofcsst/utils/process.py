import numpy as np
from ofcsst.utils import constants, ids, paths


def get_performance(outcomes: np.ndarray) -> np.ndarray:
    """Compute performance trace as a moving average of the rewarded outcomes"""
    conv_filter = 100 * np.ones(constants.PERF_MAV_N) / constants.PERF_MAV_N
    padded_outcomes = np.concatenate(
        (
            outcomes[: int(constants.PERF_MAV_N / 2)],
            outcomes,
            outcomes[-int(constants.PERF_MAV_N / 2) + 1 :],
        )
    )
    return np.convolve(a=padded_outcomes, v=conv_filter, mode="valid")


def get_expert_t(
    performances: np.ndarray, simulation: bool = True
) -> float | np.ndarray:
    """Return first trial reaching expert performance.
    - If `performances` is 1D: returns a scalar (index or np.nan)
    - If `performances` is 2D: returns a 1D array (one per row) with indices or np.nan
    """
    performances = np.asarray(performances)
    if simulation:
        expert_mask = performances > constants.EXPERT_PERFORMANCE
    else:
        expert_mask = performances >= constants.EXPERT_D

    if performances.ndim == 1:
        if expert_mask.any():
            idx = int(np.argmax(expert_mask))
            return idx if simulation else idx + 1
        else:
            return np.nan

    elif performances.ndim == 2:
        any_reached = expert_mask.any(axis=1)  # (n_rows,)
        first_idx = expert_mask.argmax(axis=1).astype(
            float
        )  # (n_rows,) float to allow NaN
        first_idx[~any_reached] = np.nan

        if not simulation:
            first_idx[any_reached] += 1.0

        return first_idx

    else:
        raise ValueError(
            f"`performances` must be 1D or 2D, got shape {performances.shape}"
        )


def get_events(
    performances: np.ndarray, simulation: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Returns to arrays:
    - one with the index of the first observed event
    - one whether the event was reached at all.
    """

    performances = np.asarray(performances, dtype=float)
    if performances.ndim != 2:
        raise ValueError(f"`performances` must be 2D, got shape {performances.shape}")

    observed = np.isfinite(performances)

    if simulation:
        reached = observed & (performances > constants.EXPERT_PERFORMANCE)
    else:
        reached = observed & (performances >= constants.EXPERT_D)

    event = reached.any(axis=1)
    first_reach = reached.argmax(axis=1) + 1  # 1-based sessions
    last_observed = observed.sum(axis=1)

    if np.any(last_observed == 0):
        raise ValueError("Some mice have no observed performance data.")

    time = np.where(event, first_reach, last_observed).astype(float)
    return time, event.astype(int)


def get_mouse_performances(
    mouse_type: ids.MouseType, learning_rule: ids.LearningRule
) -> np.ndarray:
    return np.genfromtxt(
        paths.EXPERIMENT_DIR / f"perf_{mouse_type}_{learning_rule}.csv",
        delimiter=",",
        dtype=float,
        filling_values=np.nan,
    )
