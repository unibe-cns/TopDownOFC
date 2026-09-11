"""Construction and execution of a single simulated agent.

`init_simulation` is the one dispatch point from a `SimulationType` plus a flat
list of parameter strings to a concrete agent, task and logger. The parameter
order is positional and must match `databases.get_param_keys` for the same
simulation type. `run_simulation` then drives one agent through a sequence of
trial blocks, reversing the task rule between them.
"""
import random
import torch
import numpy as np
from typing import List, Tuple, Any
from numpy import dtype, ndarray
from ofcsst.simulation import agents
from ofcsst.utils import ids, constants
from ofcsst.simulation import network, tasks

# Sentinel written into result arrays before a trial fills them in, so an
# unfilled slot is obvious in the database rather than reading as a real value.
NO_VALUE = -666.666


def init_simulation(
    task_id: ids.TaskID,
    simulation_type: ids.SimulationType,
    params: List[str],
    trial_nrs: tuple,
    log_type: ids.LogType | None = None,
) -> Tuple[agents.Agent, tasks.Task2vN, list]:
    """Build the agent, task and logger arrays for one simulation.

    :param task_id: which task environment to instantiate.
    :param simulation_type: which model variant to build.
    :param params: hyperparameter values as strings, read positionally. The first
        four are always distractor amplitude, signal amplitude, number of
        distractors and number of signals; the rest are the model's own
        hyperparameters in the order given by `databases.get_param_keys` for this
        simulation type.
    :param trial_nrs: number of trials per context, used to size the log arrays.
    :param log_type: which set of internal variables to record, or None for none.
    :returns: the agent, the task and the (possibly empty) list of log arrays.
    """
    distractor_amplitude = float(params[0])
    signal_amplitude = float(params[1])
    nr_noise = int(params[2])
    nr_signal = int(params[3])
    tot_nr_neurons = nr_noise + nr_signal
    if simulation_type in [ids.SIM_PG_NO_GAIN]:
        if len(params) > 5.5:
            optim = ids.get_optimizer(params[5])
        else:
            optim = ids.OP_SGD
        actor = network.PGActor(
            input_size=tot_nr_neurons,
            learning_rate=float(params[4]),
            habit=False,
            optim=optim,
        )
        agent = agents.AgentNoGain(actor=actor)

    elif simulation_type in [ids.SIM_PG_NO_GAIN_DEEP]:
        actor = network.PGActorDeep(
            input_size=tot_nr_neurons, learning_rate=float(params[4])
        )
        agent = agents.AgentNoGain(actor=actor)

    elif simulation_type in [ids.SIM_PG_GAIN]:
        # Parse parameters
        policy_lr = float(params[4])
        td_learning_rate = float(params[5])

        # Gain modulator init
        gain_modulator = network.GainModulatorV(
            representation_size=tot_nr_neurons, v_lr=td_learning_rate
        )

        # Actor init
        actor = network.PGActor(
            input_size=tot_nr_neurons, learning_rate=policy_lr, habit=False
        )

        # Combine actor and gain modulator into agent
        agent = agents.AgentGain(actor=actor, gain_modulator=gain_modulator)

    elif simulation_type in [ids.SIM_PURE_PG_GAIN]:
        actor = network.PurePGActor(
            input_size=tot_nr_neurons, learning_rate=float(params[4])
        )
        agent = agents.AgentNoGain(actor=actor)

    elif simulation_type in [ids.SIM_PG_RW]:
        policy_lr = float(params[4])
        rw_lr = float(params[5])
        gain_modulator = network.GainModulatorRW(
            representation_size=tot_nr_neurons, beta=rw_lr
        )
        actor = network.PGActor(
            input_size=tot_nr_neurons, learning_rate=policy_lr, habit=False
        )
        agent = agents.AgentGain(actor=actor, gain_modulator=gain_modulator)

    elif simulation_type in [ids.SIM_SQ_NO_GAIN]:
        actor = network.QActor(
            input_size=tot_nr_neurons,
            learning_rate=float(params[4]),
        )
        agent = agents.AgentNoGain(actor=actor)

    elif simulation_type in [ids.SIM_SQ_GAIN]:
        q_lr = float(params[4])
        td_learning_rate = float(params[5])
        gain_modulator = network.GainModulatorV(
            representation_size=tot_nr_neurons, v_lr=td_learning_rate
        )
        actor = network.QActor(
            input_size=tot_nr_neurons,
            learning_rate=q_lr,
        )
        agent = agents.AgentGain(actor=actor, gain_modulator=gain_modulator)

    elif simulation_type in [ids.SIM_PURE_SQ_GAIN]:
        actor = network.PureQActor(
            input_size=tot_nr_neurons,
            learning_rate=float(params[4]),
            learning_rate_gain=float(params[4]),
        )
        agent = agents.AgentNoGain(
            actor=actor
        )  # The gain is implemented directly in the actor

    elif simulation_type in [ids.SIM_CONVEX_NO_GAIN]:
        pg_actor = network.PGActor(
            input_size=tot_nr_neurons, learning_rate=float(params[4])
        )
        q_actor = network.QActor(
            input_size=tot_nr_neurons,
            learning_rate=float(params[5]),
            temperature=constants.TEMPERATURE,
        )
        agent = agents.ConvexAgent(habit_actor=pg_actor, goal_actor=q_actor)

    elif simulation_type in [
        ids.SIM_CONVEX_GAIN,
        ids.SIM_CONVEX_GAIN_REV_RESET,
        ids.SIM_CONVEX_FAKE_OFC,
    ]:
        pg_actor = network.PGActor(
            input_size=tot_nr_neurons, learning_rate=float(params[4])
        )
        q_actor = network.QActor(
            input_size=tot_nr_neurons,
            learning_rate=float(params[5]),
            temperature=constants.TEMPERATURE,
        )
        gain_modulator = network.GainModulatorV(
            representation_size=tot_nr_neurons, v_lr=float(params[6])
        )
        if simulation_type == ids.SIM_CONVEX_GAIN:
            agent = agents.ConvexAgentGain(
                habit_actor=pg_actor, goal_actor=q_actor, gain_modulator=gain_modulator
            )
        elif simulation_type == ids.SIM_CONVEX_GAIN_REV_RESET:
            agent = agents.ConvexAgentRevReset(
                habit_actor=pg_actor, goal_actor=q_actor, gain_modulator=gain_modulator
            )
        elif simulation_type == ids.SIM_CONVEX_FAKE_OFC:
            agent = agents.ConvexAgentFakeOFC(
                habit_actor=pg_actor, goal_actor=q_actor, gain_modulator=gain_modulator
            )
        else:
            raise NotImplementedError(simulation_type)

    elif simulation_type in [
        ids.SIM_CONVEX_OFC,
        ids.SIM_OFC_XAP_SWITCH,
        ids.SIM_CONVEX_OFC_VIP_KO,
    ]:
        ofc_multi_const = float(params[4])
        ofc_thresh = float(params[5])
        policy_lr = float(params[6])
        q_lr = float(params[7])
        td_learning_rate = float(params[8])
        actor = network.PGActor(input_size=tot_nr_neurons, learning_rate=policy_lr)

        if simulation_type in [ids.SIM_CONVEX_OFC, ids.SIM_CONVEX_OFC_VIP_KO]:
            vip_ko = simulation_type == ids.SIM_CONVEX_OFC_VIP_KO
            gain_modulator = network.GainModulatorOFCSST(
                representation_size=tot_nr_neurons,
                v_lr=td_learning_rate,
                ofc_lr=q_lr,
                multi_const=ofc_multi_const,
                ofc_threshold=ofc_thresh,
                vip_ko=vip_ko,
            )
            agent = agents.ConvexAgentOFC(
                habit_actor=actor, gain_modulator=gain_modulator
            )

        elif simulation_type == ids.SIM_OFC_XAP_SWITCH:
            gain_modulator = network.GainModulatorOFCSST(
                representation_size=tot_nr_neurons,
                v_lr=td_learning_rate,
                ofc_lr=q_lr,
                multi_const=ofc_multi_const,
                ofc_threshold=ofc_thresh,
            )
            agent = agents.AgentContextualXAP(
                habit_actor=actor, gain_modulator=gain_modulator
            )

        else:
            raise NotImplementedError(simulation_type)
    else:
        raise NotImplementedError(simulation_type)

    # Init task environment
    task = tasks.get_task(
        task_id=task_id,
        nr_distractor=nr_noise,
        nr_signal=nr_signal,
        distractor_amplitude=distractor_amplitude,
        signal_amplitude=signal_amplitude,
    )

    # Number of per-trial series each log type records; the agent's log()
    # fills them in the order documented on that method.
    n_logged = {
        ids.LT_RGAIN: 2,
        ids.LT_GAIN: 4,
        ids.LT_OFC: 5,
        ids.LT_CONVEX: 6,
        ids.LT_ALL: 7,
        # LT_AGENT also records the two apical gains for top-down models
        ids.LT_AGENT: 5 if simulation_type in ids.TD_LR_SIMULATIONS else 2,
    }
    if log_type is None:
        logger = []
    elif log_type in n_logged:
        logger = [
            np.array([NO_VALUE] * sum(trial_nrs)) for _ in range(n_logged[log_type])
        ]
    else:
        raise NotImplementedError(log_type)

    return agent, task, logger


def run_simulation(
    task_id: ids.TaskID,
    simulation_type: ids.SimulationType,
    params: List[str],
    seed: int = 0,
    number_contexts: int = 2,
    number_trials: tuple | None = None,
    log_type: ids.LogType | None = None,
) -> tuple[ndarray[Any, dtype[Any]], tuple[Any, ...] | None]:
    """Run one agent through a sequence of trial blocks and return its outcomes.

    The task rule is reversed between consecutive blocks, so `number_contexts=2`
    is one learning phase followed by one reversal.
    """
    # Set pseudorandom seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    number_trials = number_trials or tuple([constants.NR_TRIALS] * number_contexts)
    corrects = np.array([NO_VALUE] * sum(number_trials))

    agent, task, logger = init_simulation(
        task_id=task_id,
        simulation_type=simulation_type,
        params=params,
        log_type=log_type,
        trial_nrs=number_trials,
    )

    for reversal_idx in range(number_contexts):
        # Set current rule of the task
        task.set_task(reversal=reversal_idx % 2 != 0)

        # Special cases
        if simulation_type in ids.FAKE_OFC_SIMULATIONS and reversal_idx != 0:
            assert isinstance(
                agent, agents.ConvexAgentRevReset | agents.ConvexAgentFakeOFC
            )
            agent.fake_ofc()

        for trial in range(number_trials[reversal_idx]):
            trial_id = sum(number_trials[:reversal_idx]) + trial
            representation = task.init_stimuli()
            action = agent.get_action(basal_input=representation)
            reward, correct = task.get_outcome(acted_upon=action)
            agent.update(reward=reward)
            corrects[trial_id] = correct
            if log_type is not None:
                logger = agent.log(logger=logger, i=trial_id, log_type=log_type)

    if logger is not None:
        logger = tuple(logger)

    return corrects, logger


def simulate_params(
    task_id: ids.TaskID,
    simulation_type: ids.SimulationType,
    params: List[str],
    seed: int = 0,
    nr_contexts: int = 2,
    n_trials: tuple | None = None,
) -> np.ndarray:
    """Run one simulation and return only the per-trial correctness."""
    correct, _ = run_simulation(
        task_id=task_id,
        simulation_type=simulation_type,
        params=params,
        seed=seed,
        number_contexts=nr_contexts,
        number_trials=n_trials,
    )

    return correct
