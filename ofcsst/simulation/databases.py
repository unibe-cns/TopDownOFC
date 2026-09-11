"""Column layouts shared by every simulation and scan database.

Each scan database holds three tables: `data` (per-trial outcomes for one seed),
`scan_evaluations` (metrics aggregated over seeds for one parameter combination)
and `best_params` (the winning combination per condition).

`get_param_keys` defines, per simulation type, the ordered list of hyperparameters
stored in those tables. That order is the contract with the positional `params`
list parsed by `simulate.init_simulation`, so the two must be changed together.
"""
import sqlite3
from typing import Tuple
from ofcsst.utils import ids, keys, sql

DATA_TABLE = "data"
SCAN_TABLE = "scan_evaluations"
BEST_PARAM_TABLE = "best_params"
ALL_TABLES = [DATA_TABLE, SCAN_TABLE, BEST_PARAM_TABLE]


def get_param_keys(simulation_type: ids.SimulationType) -> list[keys.Key]:
    """Ordered hyperparameter columns for one simulation type.

    The order is the contract with `simulate.init_simulation`, which reads the
    same values positionally out of its `params` list, and with
    `parameters.job_to_sim_params`, which writes them in this order.
    """
    if simulation_type in ids.OFC_SST_SIMULATIONS:
        ofc_keys: list[keys.Key] = [keys.SST_W, keys.SST_B]
    else:
        ofc_keys: list[keys.Key] = []

    if simulation_type in [
        ids.SIM_PG_NO_GAIN,
        ids.SIM_PURE_PG_GAIN,
        ids.SIM_PG_NO_GAIN_DEEP,
    ]:
        lr_keys: list[keys.Key] = [keys.LEARNING_RATE_PG]
    elif simulation_type in (ids.SIM_PG_GAIN, ids.SIM_PG_RW):
        lr_keys: list[keys.Key] = [keys.LEARNING_RATE_PG, keys.LEARNING_RATE_V]
    elif simulation_type in (ids.SIM_SQ_NO_GAIN, ids.SIM_PURE_SQ_GAIN):
        lr_keys: list[keys.Key] = [keys.LEARNING_RATE_Q]
    elif simulation_type in (ids.SIM_SQ_GAIN,):
        lr_keys: list[keys.Key] = [keys.LEARNING_RATE_Q, keys.LEARNING_RATE_V]
    elif simulation_type in (ids.SIM_CONVEX_NO_GAIN,):
        lr_keys: list[keys.Key] = [keys.LEARNING_RATE_PG, keys.LEARNING_RATE_Q]
    elif simulation_type in [
        ids.SIM_CONVEX_GAIN,
        ids.SIM_CONVEX_GAIN_REV_RESET,
        ids.SIM_CONVEX_FAKE_OFC,
        ids.SIM_CONVEX_OFC,
        ids.SIM_CONVEX_OFC_VIP_KO,
    ]:
        lr_keys: list[keys.Key] = [
            keys.LEARNING_RATE_PG,
            keys.LEARNING_RATE_Q,
            keys.LEARNING_RATE_V,
        ]
    else:
        raise NotImplementedError(simulation_type)

    more_keys: list[keys.Key] = []

    return ofc_keys + lr_keys + more_keys


def get_unique_cols(simulation_type: ids.SimulationType, table: str) -> list[keys.Key]:
    """Get all the columns that uniquely identify a simulation run in the given table. This means pretty much all
    columns except the outcome columns of the simulation."""

    if table in [DATA_TABLE, SCAN_TABLE]:
        param_keys: list[keys.Key] = get_param_keys(simulation_type=simulation_type)
        if table in [DATA_TABLE]:
            param_keys.append(keys.SEED)
    elif table in [BEST_PARAM_TABLE]:
        param_keys: list[keys.Key] = []
    else:
        raise ValueError(table)

    base_keys: list[keys.Key] = [
        keys.DISTRACTOR_GAIN,
        keys.SIGNAL_GAIN,
        keys.NR_DISTRACTOR,
        keys.NR_SIGNAL,
    ]

    return base_keys + param_keys


def get_best_params(
    simulation_type: ids.SimulationType, unique_values: dict, cur: sqlite3.Cursor
) -> Tuple:
    """Best performing hyperparameter values for the condition given by `unique_values`."""
    cmd, where_tuple = sql.select_where_cmd(
        table=BEST_PARAM_TABLE,
        get_cols=get_param_keys(simulation_type=simulation_type),
        where_values=unique_values,
    )
    cur.execute(cmd, where_tuple)
    return cur.fetchone()


def get_outcome_cols(table: str, n_trials: int) -> list[str]:
    """Result columns of a table: one per trial for `data`, aggregates elsewhere."""
    if table in [DATA_TABLE]:
        return [f"{keys.PERFORMANCE}_{n}" for n in range(n_trials)]
    elif table in [BEST_PARAM_TABLE, SCAN_TABLE]:
        return [keys.PERFORMANCE, keys.EXPERT_FRAC, keys.EXPERT_TIME]
    else:
        raise NotImplementedError(table)


def get_all_cols(
    simulation_type: ids.SimulationType, table: str, n_trials: int
) -> list[str]:
    """Full column list of a table: identifying columns followed by simulation results."""
    all_cols = get_unique_cols(simulation_type=simulation_type, table=table)
    if table in [BEST_PARAM_TABLE]:
        all_cols += get_param_keys(simulation_type=simulation_type)
    all_cols += get_outcome_cols(table=table, n_trials=n_trials)
    return all_cols
