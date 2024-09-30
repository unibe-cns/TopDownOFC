import sqlite3
from typing import Tuple
from ofcsst.utils import ids, keys, sql

DATA_TABLE = "data"
SCAN_TABLE = "scan_evaluations"
BEST_PARAM_TABLE = "best_params"
ALL_TABLES = [DATA_TABLE, SCAN_TABLE, BEST_PARAM_TABLE]


def get_lr_keys(simulation_type: ids.SimulationType):
    if simulation_type in [ids.SIM_PG_NO_GAIN]:
        return [keys.LEARNING_RATE_PG]
    elif simulation_type in [ids.SIM_PG_GAIN]:
        return [keys.LEARNING_RATE_PG, keys.LEARNING_RATE_V]
    elif simulation_type in [ids.SIM_CONVEX_NO_GAIN]:
        return [keys.LEARNING_RATE_PG, keys.LEARNING_RATE_Q]
    elif simulation_type in [ids.SIM_CONVEX_GAIN, ids.SIM_CONVEX_GAIN_REV_RESET, ids.SIM_CONVEX_FAKE_OFC,
                             ids.SIM_CONVEX_OFC, ids.SIM_CONVEX_OFC_VIP_KO]:
        return [keys.LEARNING_RATE_PG, keys.LEARNING_RATE_Q, keys.LEARNING_RATE_V]
    else:
        raise NotImplementedError(simulation_type)


def get_param_keys(simulation_type: ids.SimulationType):
    if simulation_type in ids.OFC_SST_SIMULATIONS:
        ofc_keys: list[keys.Key] = [keys.SST_W, keys.SST_B]
    else:
        ofc_keys: list[keys.Key] = []

    lr_keys: list[keys.Key] = get_lr_keys(simulation_type=simulation_type)

    return ofc_keys + lr_keys


def get_unique_cols(simulation_type: ids.SimulationType, table: str) -> list[keys.Key]:
    lr_keys: list[keys.Key] = get_lr_keys(simulation_type=simulation_type)

    if simulation_type in ids.OFC_SST_SIMULATIONS:
        ofc_keys: list[keys.Key] = [keys.SST_W, keys.SST_B]
    else:
        ofc_keys: list[keys.Key] = []

    if table in [DATA_TABLE, SCAN_TABLE]:
        special_keys: list[keys.Key] = ofc_keys + lr_keys
        if table in [DATA_TABLE]:
            special_keys += [keys.SEED]
    elif table in [BEST_PARAM_TABLE]:
        special_keys: list[keys.Key] = []
    else:
        raise ValueError(table)

    base_keys: list[keys.Key] = [keys.DISTRACTOR_GAIN, keys.SIGNAL_GAIN, keys.NR_DISTRACTOR,
                                 keys.NR_SIGNAL]

    return base_keys + special_keys


def get_best_lr(simulation_type: ids.SimulationType, unique_values: dict, cur: sqlite3.Cursor) -> Tuple:
    cmd, where_tuple = sql.select_where_cmd(table=BEST_PARAM_TABLE,
                                            get_cols=get_lr_keys(simulation_type=simulation_type),
                                            where_values=unique_values)
    cur.execute(cmd, where_tuple)
    return cur.fetchone()


def get_outcome_cols(table: str, n_trials: int) -> list[str]:
    if table in [DATA_TABLE]:
        return [f"{keys.PERFORMANCE}_{n}" for n in range(n_trials)]
    elif table in [BEST_PARAM_TABLE, SCAN_TABLE]:
        return [keys.PERFORMANCE, keys.EXPERT_FRAC, keys.EXPERT_TIME]
    else:
        raise NotImplementedError(table)


def get_all_cols(simulation_type: ids.SimulationType, table: str, n_trials: int) -> list[str]:
    all_cols = get_unique_cols(simulation_type=simulation_type, table=table)
    if table in [BEST_PARAM_TABLE]:
        all_cols += tuple(get_param_keys(simulation_type=simulation_type))
    all_cols += get_outcome_cols(table=table, n_trials=n_trials)
    return all_cols
