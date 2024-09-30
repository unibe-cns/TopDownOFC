import math
import os
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, TextIO
from ofcsst.utils import ids, keys, paths, sql, constants, process
from ofcsst.simulation import databases, simulate, parameters


JOB_DIR = paths.ROOT_DIR / 'jobs'
LR_BASE = 10
LEARNING_RATES = [LR_BASE ** p for p in np.linspace(start=-2, stop=1, num=4, endpoint=True)]
NUMBER_DISTRACTORS = [2 ** p for p in range(12)]
SIGNAL_AMPLITUDES = [2 ** p for p in range(-4, 5)]
RESET_ID = 0


def get_db_path(scan_type: ids.ScanType, simulation_type: ids.SimulationType, task_id: ids.TaskID = constants.TASK_ID,
                non_stationary: bool = True, optim: ids.OptimizerType = ids.OP_SGD) -> Path:
    if non_stationary:
        s_dir = paths.SWITCH_SUBDIR
    else:
        s_dir = paths.STATIONARY_SUBDIR
    if optim == ids.OP_SGD:
        db_path = paths.RESULT_DIR / paths.SCAN_SUBDIR / simulation_type / task_id / s_dir / f"{scan_type}.db"
    else:
        db_path = paths.RESULT_DIR / paths.SCAN_SUBDIR / simulation_type / task_id / s_dir / f"{scan_type}_{optim}.db"

    return db_path


def get_job_path(scan_type: ids.ScanType, simulation_type: ids.SimulationType, task_id: ids.TaskID,
                 non_stationary: bool = True, optim: ids.OptimizerType = ids.OP_SGD) -> Path:
    if non_stationary:
        s_dir = paths.SWITCH_SUBDIR
    else:
        s_dir = paths.STATIONARY_SUBDIR
    if optim == ids.OP_SGD:
        job_path = JOB_DIR / paths.SCAN_SUBDIR / simulation_type / task_id / s_dir / f'job_{scan_type}.txt'
    else:
        job_path = JOB_DIR / paths.SCAN_SUBDIR / simulation_type / task_id / s_dir / f'job_{scan_type}_{optim}.txt'

    return job_path


def scan_forward(task_id: ids.TaskID, simulation_type: ids.SimulationType, params: List[str], conn: sqlite3.Connection,
                 cur: sqlite3.Cursor, base_insert_values: tuple, insert_cmd: dict) -> None:
    outcomes = np.zeros((constants.NR_SEEDS, constants.NR_TRIALS))
    final_performances = np.zeros((constants.NR_SEEDS,))
    frac_experts = np.zeros((constants.NR_SEEDS,))
    expert_times = np.zeros((constants.NR_SEEDS,))
    for seed in range(constants.NR_SEEDS):
        try:
            outcomes[seed, :] = simulate.simulate_params(task_id=task_id, simulation_type=simulation_type, seed=seed,
                                                         params=params, nr_contexts=1)
        except ValueError:
            outcomes[seed, :] = np.array([-1.] * constants.NR_TRIALS)
        performance = process.get_performance(outcomes=outcomes[seed, :])
        final_performances[seed] = performance[-1]
        frac_experts[seed] = int(np.any(performance > constants.EXPERT_PERFORMANCE))
        expert_times[seed] = process.get_expert_t(performances=performance)
        insert_values = base_insert_values + (seed,)
        cur.execute(insert_cmd[databases.DATA_TABLE], insert_values + tuple(outcomes[seed, :]))
    if np.mean(frac_experts) == 1:
        expert_time = np.nanmean(expert_times)
    else:
        expert_time = 3 * constants.NR_TRIALS
    insert_values = base_insert_values + (np.mean(final_performances), np.mean(frac_experts), expert_time)
    cur.execute(insert_cmd[databases.SCAN_TABLE], insert_values)
    conn.commit()


def scan_reversal(task_id: ids.TaskID, simulation_type: ids.SimulationType, params: List[str], conn: sqlite3.Connection,
                  cur: sqlite3.Cursor, base_insert_values: tuple, insert_cmd: dict, n_trials: tuple = None) -> None:
    if n_trials is None:
        trial_0_rev = constants.NR_TRIALS
    else:
        trial_0_rev = n_trials[0]
    reversal_performances = np.zeros((constants.NR_SEEDS,))
    frac_experts = np.zeros((constants.NR_SEEDS,))
    expert_times = np.zeros((constants.NR_SEEDS,))
    for seed in range(constants.NR_SEEDS):
        try:
            outcomes = simulate.simulate_params(task_id=task_id, simulation_type=simulation_type, seed=seed,
                                                params=params, nr_contexts=2, n_trials=n_trials)
        except ValueError:
            outcomes = np.array([-666.666] * constants.NR_TRIALS * 2)

        performance = process.get_performance(outcomes=outcomes[trial_0_rev:])
        reversal_performances[seed] = performance[-1]
        frac_experts[seed] = int(np.any(performance > constants.EXPERT_PERFORMANCE))
        expert_times[seed] = process.get_expert_t(performances=performance)
    if np.mean(frac_experts) == 1:
        expert_time = np.nanmean(expert_times)
    else:
        expert_time = 3 * constants.NR_TRIALS
    insert_values = base_insert_values + (np.mean(reversal_performances), np.mean(frac_experts), expert_time)
    cur.execute(insert_cmd[databases.SCAN_TABLE], insert_values)
    conn.commit()


def scan_params(simulation_type: ids.SimulationType, task_id: ids.TaskID, scan_type: ids.ScanType,
                switch_task: bool = True, optim: ids.OptimizerType = ids.OP_SGD) -> None:
    if switch_task:
        tables = [databases.SCAN_TABLE]
        if simulation_type in [ids.SIM_CONVEX_GAIN, ids.SIM_PG_NO_GAIN]:
            n_trials = 3 * constants.NR_TRIALS
            trial_nrs = (constants.NR_TRIALS, 2 * constants.NR_TRIALS)
        else:
            n_trials = 2 * constants.NR_TRIALS
            trial_nrs = (constants.NR_TRIALS, constants.NR_TRIALS)

    else:
        tables = [databases.DATA_TABLE, databases.SCAN_TABLE]
        n_trials = constants.NR_TRIALS
        trial_nrs = (constants.NR_TRIALS,)
    if simulation_type in [ids.SIM_PG_NO_GAIN]:
        param_add = [optim]
    else:
        param_add = []

    # Read jobs
    job_file = get_job_path(scan_type=scan_type, simulation_type=simulation_type, task_id=task_id,
                            non_stationary=switch_task, optim=optim)
    with open(job_file, 'r') as f:
        jobs = f.readlines()

    # Connect with the database
    db_path = get_db_path(scan_type=scan_type, simulation_type=simulation_type, task_id=task_id,
                          non_stationary=switch_task, optim=optim)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    col_keys = {}
    insert_cmd = {}
    for table in tables:
        col_keys[table]: list[str] = databases.get_all_cols(simulation_type=simulation_type, table=table,
                                                            n_trials=n_trials)
        sql.make_table(conn=conn, table_name=table, col_names=col_keys[table], verbose=False)
        insert_cmd[table] = sql.get_insert_cmd(table=table, col_keys=col_keys[table])
    cur = conn.cursor()
    progress_tot = len(jobs)
    for progress, job in enumerate(jobs):
        print(f"\rScanning {100 * progress / progress_tot:0.1f}% completed", end="")
        params = job.strip().split(';') + param_add
        base_insert_values = parameters.job_to_sim_params(simulation_type=simulation_type, params=params)
        if switch_task:
            scan_reversal(
                task_id=task_id, simulation_type=simulation_type, params=params, conn=conn, cur=cur,
                base_insert_values=base_insert_values, insert_cmd=insert_cmd, n_trials=trial_nrs
            )
        else:
            scan_forward(
                task_id=task_id, simulation_type=simulation_type, params=params, conn=conn, cur=cur,
                base_insert_values=base_insert_values, insert_cmd=insert_cmd
            )
    cur.close()
    conn.close()
    del conn
    os.remove(job_file)
    print(". Terminated successfully.")


def get_scan_params_env(scan_type: ids.ScanType = ids.ST_FINAL) -> tuple:
    noise_stds = [1.]
    nr_signals = [2]
    if scan_type in [ids.ST_VARY_DISTRACTION_2D]:
        signal_amplitudes = SIGNAL_AMPLITUDES
    else:
        signal_amplitudes = [1.]
    if scan_type in [ids.ST_VARY_DISTRACTOR_1D, ids.ST_VARY_DISTRACTION_2D]:
        nr_distract = NUMBER_DISTRACTORS
    elif scan_type == ids.ST_NO_DISTRACTOR:
        nr_distract = [0]
    else:
        nr_distract = [constants.NUMBER_DISTRACTOR]
    return noise_stds, signal_amplitudes, nr_distract, nr_signals


def write_job(text: str, param_ranges: [[float]], f: TextIO):
    if len(param_ranges) == 0:
        f.write(f'{text}\n')
    else:
        for p in param_ranges[0]:
            app_text = f'{text};{p}'
            write_job(text=app_text, param_ranges=param_ranges[1:], f=f)


def write_job_check(text: str, param_ranges: [[float]], f: TextIO, cur: sqlite3.Cursor, unique_values: dict,
                    param_keys: List, suffix: str = ''):
    if len(param_ranges) == 0:
        unique_cols = list(unique_values.keys())
        if not sql.row_exists(cur=cur, table=databases.SCAN_TABLE, unique_values=unique_values,
                              unique_cols=unique_cols):
            f.write(f'{text}{suffix}\n')
    else:
        for p in param_ranges[0]:
            app_text = f'{text};{p}'
            unique_values[param_keys[0]] = p
            write_job_check(text=app_text, param_ranges=param_ranges[1:], f=f, cur=cur, unique_values=unique_values,
                            param_keys=param_keys[1:], suffix=suffix)


def write_lr_jobs(sim_type: ids.SimulationType, id_lr: int, noise_std: float, signal_amplitude: float, nr_noise: int,
                  nr_signal: int, f: TextIO, cur: sqlite3.Cursor, continued: bool = False) -> None:

    text = f'{noise_std:.3e};{signal_amplitude:.3e};{nr_noise};{nr_signal}'
    lr_keys = databases.get_lr_keys(simulation_type=sim_type)
    unique_values = {keys.DISTRACTOR_GAIN: noise_std, keys.SIGNAL_GAIN: signal_amplitude,
                     keys.NR_DISTRACTOR: nr_noise, keys.NR_SIGNAL: nr_signal}

    if id_lr == 0:
        lr_ranges = [LEARNING_RATES for _ in databases.get_lr_keys(simulation_type=sim_type)]
        if continued:
            write_job_check(text=text, param_ranges=lr_ranges, f=f, cur=cur, unique_values=unique_values,
                            param_keys=lr_keys)
        else:
            write_job(text=text, param_ranges=lr_ranges, f=f)
    else:
        surrounds = [float(i - 2) / (2. ** id_lr) for i in range(5)]
        lr_opts = databases.get_best_lr(simulation_type=sim_type, unique_values=unique_values, cur=cur)
        lr_ranges = []
        for lr_opt in lr_opts:
            b_lr_p = round(math.log(lr_opt, LR_BASE), 5)
            lr_ranges += [[LR_BASE ** (b_lr_p + p) for p in surrounds]]
        write_job_check(text=text, param_ranges=lr_ranges, f=f, cur=cur, unique_values=unique_values,
                        param_keys=lr_keys)


def write_scan_jobs_default(f, simulation_type: ids.SimulationType, scan_type: ids.ScanType, switch_task: bool,
                            task_id: ids.TaskID, id_lr: int, optim: ids.OptimizerType = ids.OP_SGD):
    if switch_task:
        sub_rev = paths.SWITCH_SUBDIR
        n_rev = 2
    else:
        sub_rev = paths.STATIONARY_SUBDIR
        n_rev = 1
    n_trials = n_rev * constants.NR_TRIALS
    print(f"Scanning {scan_type} {simulation_type} on {sub_rev} {optim} round {id_lr}")

    # Write a job file with all the parameter configurations to scan according to the scan type.
    # Init database variables
    db_path = get_db_path(scan_type=scan_type, simulation_type=simulation_type, task_id=task_id,
                          non_stationary=switch_task, optim=optim)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    for table in databases.ALL_TABLES:
        if id_lr in [RESET_ID]:
            sql.drop_table(conn=conn, table_name=table, verbose=False)
        cols = databases.get_all_cols(simulation_type=simulation_type, table=table, n_trials=n_trials)
        sql.make_table(conn=conn, table_name=table, col_names=cols, verbose=False)
    cur = conn.cursor()

    noise_stds, signal_amplitudes, nr_noises, nr_signals = get_scan_params_env(
        scan_type=scan_type
    )
    for noise_std in noise_stds:
        for signal_amplitude in signal_amplitudes:
            for nr_noise in nr_noises:
                for nr_signal in nr_signals:
                    if scan_type in [ids.ST_VARY_DISTRACTOR_1D, ids.ST_VARY_DISTRACTION_2D, ids.ST_FINAL,
                                     ids.ST_NO_DISTRACTOR]:
                        write_lr_jobs(sim_type=simulation_type, id_lr=id_lr, noise_std=noise_std,
                                      signal_amplitude=signal_amplitude, nr_noise=nr_noise, nr_signal=nr_signal, f=f,
                                      cur=cur)
                    else:
                        raise NotImplementedError(scan_type)

    cur.close()
    conn.close()


def write_scan_jobs_ofc_all(f, task_id: ids.TaskID, id_lr: int):
    print(f"Scanning {ids.ST_ALL} {ids.SIM_CONVEX_OFC} on {paths.SWITCH_SUBDIR} {task_id} round {id_lr}")

    # Get other params
    noise_stds, signal_amplitudes, nr_noises, nr_signals = get_scan_params_env(
        scan_type=ids.ST_FINAL
    )

    # Init database
    db_path = get_db_path(scan_type=ids.ST_ALL, simulation_type=ids.SIM_CONVEX_OFC, task_id=task_id,
                          non_stationary=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    for table in databases.ALL_TABLES:
        if id_lr == RESET_ID:
            sql.drop_table(conn=conn, table_name=table, verbose=False)
        cols = databases.get_all_cols(simulation_type=ids.SIM_CONVEX_OFC, table=table, n_trials=2 * constants.NR_TRIALS)
        sql.make_table(conn=conn, table_name=table, col_names=cols, verbose=False)

    # Get optimal learning rates from previous optimization
    param_keys = [keys.SST_W, keys.SST_B, keys.LEARNING_RATE_PG, keys.LEARNING_RATE_Q, keys.LEARNING_RATE_V]
    if id_lr == 0:
        scan_db_path = get_db_path(scan_type=ids.ST_TD, simulation_type=ids.SIM_CONVEX_OFC, task_id=task_id,
                                   non_stationary=True)
        best_params = sql.get_max(db_path=scan_db_path, table=databases.BEST_PARAM_TABLE, max_col=keys.PERFORMANCE,
                                  group_cols=[], select_cols=param_keys)[0]
    else:
        best_params = sql.get_max(db_path=db_path, table=databases.BEST_PARAM_TABLE, max_col=keys.PERFORMANCE,
                                  group_cols=[], select_cols=param_keys)[0]

    cur = conn.cursor()
    lr_ranges = []
    surrounds = [float(i - 1) / (2. ** id_lr) for i in range(3)]
    unique_values = {keys.DISTRACTOR_GAIN: noise_stds[0], keys.SIGNAL_GAIN: signal_amplitudes[0],
                     keys.NR_DISTRACTOR: nr_noises[0], keys.NR_SIGNAL: nr_signals[0]}
    for best_param in best_params:
        b_lr_p = round(math.log(best_param, LR_BASE), 5)
        lr_ranges += [[LR_BASE ** (b_lr_p + p) for p in surrounds]]
    text = f'{noise_stds[0]:.3e};{signal_amplitudes[0]:.3e};{nr_noises[0]};{nr_signals[0]}'
    write_job_check(text=text, param_ranges=lr_ranges, f=f, cur=cur, unique_values=unique_values,
                    param_keys=param_keys)
    cur.close()
    conn.close()


def write_scan_jobs_td(f, task_id: ids.TaskID, id_lr: int):
    print(f"Scanning {ids.ST_TD} {ids.SIM_CONVEX_OFC} on {paths.SWITCH_SUBDIR} {task_id} round {id_lr}")

    # Get other params
    noise_stds, signal_amplitudes, nr_noises, nr_signals = get_scan_params_env(
        scan_type=ids.ST_FINAL
    )

    # Init database
    db_path = get_db_path(scan_type=ids.ST_TD, simulation_type=ids.SIM_CONVEX_OFC, task_id=task_id,
                          non_stationary=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    for table in databases.ALL_TABLES:
        if id_lr == 0:
            sql.drop_table(conn=conn, table_name=table, verbose=False)
        cols = databases.get_all_cols(simulation_type=ids.SIM_CONVEX_OFC, table=table, n_trials=2 * constants.NR_TRIALS)
        sql.make_table(conn=conn, table_name=table, col_names=cols, verbose=False)

    # Get optimal learning rates from previous optimization
    scan_db_path = get_db_path(scan_type=ids.ST_FINAL, simulation_type=ids.SIM_CONVEX_GAIN, task_id=task_id,
                               non_stationary=True)
    scan_conn = sqlite3.connect(scan_db_path)
    scan_cur = scan_conn.cursor()
    try:
        p_lr, q_lr, td_lr = databases.get_best_lr(simulation_type=ids.SIM_CONVEX_GAIN, unique_values={}, cur=scan_cur)
    except TypeError or sqlite3.OperationalError:
        p_lr, q_lr, td_lr = 10. ** 1.25, 10., 10. ** 1.25
    scan_cur.close()
    scan_conn.close()

    if id_lr == 0:
        # Parameters to scan
        mult_consts = np.power(10, np.linspace(0.5, 2, 4, endpoint=True))
        thresholds = np.power(10, np.linspace(-1.25, -0.5, 4, endpoint=True))

        # Write jobs for each parameter configuration
        for mult_const in mult_consts:
            for threshold in thresholds:
                f.write(f'{noise_stds[0]:.3e};{signal_amplitudes[0]:.3e};{nr_noises[0]};{nr_signals[0]};'
                        f'{mult_const};{threshold};{p_lr};{q_lr};{td_lr}\n')
    else:
        surrounds = [float(i - 1) / (2. ** id_lr) for i in range(3)]
        param_keys = [keys.SST_W, keys.SST_B, keys.LEARNING_RATE_V]
        best_params = sql.get_max(db_path=db_path, table=databases.BEST_PARAM_TABLE, max_col=keys.PERFORMANCE,
                                  group_cols=[], select_cols=param_keys)[0]
        cur = conn.cursor()

        unique_values = {keys.DISTRACTOR_GAIN: noise_stds[0], keys.SIGNAL_GAIN: signal_amplitudes[0],
                         keys.NR_DISTRACTOR: nr_noises[0], keys.NR_SIGNAL: nr_signals[0], keys.LEARNING_RATE_PG: p_lr,
                         keys.LEARNING_RATE_Q: q_lr}
        lr_ranges = []
        for best_param in best_params:
            b_lr_p = round(math.log(best_param, LR_BASE), 5)
            lr_ranges += [[LR_BASE ** (b_lr_p + p) for p in surrounds]]
        text = f'{noise_stds[0]:.3e};{signal_amplitudes[0]:.3e};{nr_noises[0]};{nr_signals[0]}'
        for lr_td in lr_ranges[2]:
            suffix = f';{p_lr};{q_lr};{lr_td}'
            unique_values[param_keys[2]] = lr_td
            write_job_check(text=text, param_ranges=lr_ranges[:2], f=f, cur=cur, unique_values=unique_values,
                            param_keys=param_keys[:2], suffix=suffix)
        cur.close()
    conn.close()


def write_scan_jobs(simulation_type: ids.SimulationType, scan_type: ids.ScanType, task_id: ids.TaskID,
                    id_lr: int = 0, switch_task: bool = False, optim: ids.OptimizerType = ids.OP_SGD) -> bool:
    # Load scan jobs
    job_file = get_job_path(scan_type=scan_type, simulation_type=simulation_type, task_id=task_id,
                            non_stationary=switch_task, optim=optim)
    job_file.parent.mkdir(parents=True, exist_ok=True)

    # Run jobs
    with open(job_file, 'w') as f:
        if scan_type == ids.ST_TD:
            write_scan_jobs_td(f=f, task_id=task_id, id_lr=id_lr)
        elif scan_type == ids.ST_ALL:
            write_scan_jobs_ofc_all(f=f, task_id=task_id, id_lr=id_lr)
        else:
            write_scan_jobs_default(f=f, simulation_type=simulation_type, scan_type=scan_type, switch_task=switch_task,
                                    task_id=task_id, id_lr=id_lr, optim=optim)

    return os.stat(job_file).st_size != 0


def update_best_lrs(simulation_type: ids.SimulationType, task_id: ids.TaskID, scan_type: ids.ScanType = ids.ST_FINAL,
                    switch_task: bool = True, optim: ids.OptimizerType = ids.OP_SGD) -> None:

    scan_table = databases.SCAN_TABLE
    best_table = databases.BEST_PARAM_TABLE

    if switch_task:
        n_trials = 2 * constants.NR_TRIALS
    else:
        n_trials = constants.NR_TRIALS

    # Define column names for db operations
    best_cols = databases.get_all_cols(simulation_type=simulation_type, table=best_table, n_trials=n_trials)
    group_cols = databases.get_unique_cols(simulation_type=simulation_type, table=best_table)

    # Init db connection
    db_path = get_db_path(scan_type=scan_type, simulation_type=simulation_type, task_id=task_id,
                          non_stationary=switch_task, optim=optim)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Update best lr table
    sql.drop_table(conn=conn, table_name=best_table, verbose=False)
    sql.make_table(conn=conn, table_name=best_table, col_names=best_cols)

    if scan_type in [ids.ST_FINAL, ids.ST_ALL, ids.ST_TD, ids.ST_NO_DISTRACTOR]:
        n_success = sql.count_rows(cur=cur, table=scan_table, unique_values={keys.EXPERT_FRAC: 1},
                                   unique_cols=[keys.EXPERT_FRAC])
        if n_success >= 1:
            # If some parameter configurations produce 100% expert, select the parameters that learn the fastest
            best_train_lrs = sql.get_max(db_path=db_path, table=scan_table, group_cols=group_cols,
                                         select_cols=best_cols, max_col=keys.EXPERT_TIME, maxmin=False)
            print(f"The best expert time is {best_train_lrs[0][-1]}")

        else:
            # Otherwise, select the parameters with the highest average performance
            best_train_lrs = sql.get_max(db_path=db_path, table=scan_table, group_cols=group_cols,
                                         select_cols=best_cols, max_col=keys.PERFORMANCE, maxmin=True)
            print(f"The best performance is {best_train_lrs[0][-3]}")
    elif scan_type in [ids.ST_VARY_DISTRACTOR_1D, ids.ST_VARY_DISTRACTION_2D]:
        best_train_lrs = sql.get_max(db_path=db_path, table=scan_table, group_cols=group_cols, select_cols=best_cols,
                                     max_col=keys.PERFORMANCE, maxmin=True)
        print(f"The best performance is {best_train_lrs[0][-3]}")
    else:
        raise ValueError(scan_type)

    insert_cmd = sql.get_insert_cmd(table=best_table, col_keys=best_cols)

    for best_params_set in best_train_lrs:
        cur.execute(insert_cmd, best_params_set)
    conn.commit()
    cur.close()
    conn.close()


def find_best_params_id(simulation_type: ids.SimulationType, task_id: ids.TaskID,
                        scan_type: ids.ScanType = ids.ST_FINAL, switch_task: bool = False, id_lr: int = 0,
                        optim: ids.OptimizerType = ids.OP_SGD) -> bool:
    """
    Define the set of parameters to simulate, write these sets down in a job file, simulate the parameters according to
    the job file and finally check which set of parameters is the best.
    :param simulation_type: ID of the model type
    :param task_id: ID of the task
    :param scan_type: Type of scan, encoding the parameters to optimize and the environmental variables to vary
    :param switch_task: Whether the task will be changed during the simulation
    :param id_lr: Round of scan to perform
    :param optim: Type of optimizer
    :return: Returns true if some new parameters were tested and false if the round has already tested all necessary
    sets of parameters.
    """
    written = write_scan_jobs(id_lr=id_lr, simulation_type=simulation_type, task_id=task_id,
                              switch_task=switch_task, scan_type=scan_type, optim=optim)
    if written:
        scan_params(simulation_type=simulation_type, task_id=task_id, switch_task=switch_task, scan_type=scan_type,
                    optim=optim)
        update_best_lrs(simulation_type, task_id=task_id, switch_task=switch_task, scan_type=scan_type, optim=optim)
    return written


def find_best_params_loop(simulation_type: ids.SimulationType, task_id: ids.TaskID,
                          scan_type: ids.ScanType = ids.ST_FINAL, switch_task: bool = False, id_lr: int = 0,
                          optim: ids.OptimizerType = ids.OP_SGD) -> None:
    """
    Run one round of parameter scan. Either run the initial round with a fixed grid, or run one of the next rounds where
    the grid is iteratively defined around the best set of parameters according to a resolution that decreases
    exponentially with each round. The grid is tested as long as there are untested parameter configurations surrounding
    the best set of parameters. Once the surroundings have been tested and the best set of parameters is clear, the
    round ends.
    :param simulation_type: ID of the model type
    :param task_id: ID of the task
    :param scan_type: Type of scan, encoding the parameters to optimize and the environmental variables to vary
    :param switch_task: Whether the task will be changed during the simulation
    :param id_lr: Round of scan to perform
    :param optim: Type of optimizer
    """
    if id_lr == 0:
        find_best_params_id(
            simulation_type=simulation_type, task_id=task_id, scan_type=scan_type, switch_task=switch_task,
            id_lr=id_lr, optim=optim
        )
    else:
        more_params_to_try = True
        while more_params_to_try:
            more_params_to_try = find_best_params_id(
                simulation_type=simulation_type, task_id=task_id, scan_type=scan_type, switch_task=switch_task,
                id_lr=id_lr, optim=optim
            )


def find_best_params(simulation_type: ids.SimulationType, task_id: ids.TaskID,
                     scan_type: ids.ScanType = ids.ST_FINAL, switch_task: bool = False, id_start: int = 0,
                     nr_id_lr: int = 3, optim: ids.OptimizerType = ids.OP_SGD) -> None:
    """
    Run iterative rounds of grid parameter scans, where the next round of scan will be centered around the best set of
    parameters from the last round. The parameter grids are set exponentially, meaning uniformly along powers of 10.
    Each new round of scan narrows the search grid down to the square root of the previous resolution.
    :param simulation_type: ID of the model type
    :param task_id: ID of the task
    :param scan_type: Type of scan, encoding the parameters to optimize and the environmental variables to vary
    :param switch_task: Whether the task will be changed during the simulation
    :param id_start: Scan round to start from (should be zero unless one wishes to continue a scan that was interrupted)
    :param nr_id_lr: Final number of parameter scan rounds to run
    :param optim: Type of optimizer
    """

    for id_lr in range(id_start, nr_id_lr):
        find_best_params_loop(simulation_type=simulation_type, task_id=task_id, scan_type=scan_type,
                              switch_task=switch_task, id_lr=id_lr, optim=optim)
