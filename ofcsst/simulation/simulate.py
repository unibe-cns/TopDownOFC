import random
import torch
import numpy as np
from typing import List, Tuple
from ofcsst.simulation import agents
from ofcsst.utils import ids, constants
from ofcsst.simulation import network, tasks

REWARD = torch.Tensor([1.])
PUNISHMENT = torch.Tensor([-1.])
NO_VALUE = -666.666


def init_simulation(task_id: ids.TaskID, simulation_type: ids.SimulationType, params: List[str], trial_nrs: tuple,
                    log_type: ids.LogType = None) -> Tuple[agents.AgentTypes, tasks.Task2vN, list]:
    distractor_amplitude = float(params[0])
    signal_amplitude = float(params[1])
    nr_noise = int(params[2])
    nr_signal = int(params[3])
    tot_nr_neurons = nr_noise + nr_signal
    if simulation_type in [ids.SIM_PG_NO_GAIN]:
        if len(params) > 5.5:
            optim = ids.get_literal(params[5])
        else:
            optim = ids.OP_SGD
        actor = network.PGActor(input_size=tot_nr_neurons, learning_rate=float(params[4]), habit=False, optim=optim)
        agent = agents.PGAgentNoGain(actor=actor)

    elif simulation_type in [ids.SIM_PG_GAIN]:

        # Parse parameters
        policy_lr = float(params[4])
        td_learning_rate = float(params[5])

        # Gain modulator  init
        gain_modulator = network.GainModulator(representation_size=tot_nr_neurons, v_lr=td_learning_rate)

        # Actor init
        actor = network.PGActor(input_size=tot_nr_neurons, learning_rate=policy_lr, habit=False)

        # Combine actor and gain modulator into agent
        agent = agents.PGAgentGain(actor=actor, gain_modulator=gain_modulator)

    elif simulation_type in [ids.SIM_CONVEX_NO_GAIN]:
        pg_actor = network.PGActor(input_size=tot_nr_neurons, learning_rate=float(params[4]))
        q_actor = network.QActor(input_size=tot_nr_neurons, learning_rate=float(params[5]))
        agent = agents.ConvexAgent(habit_actor=pg_actor, goal_actor=q_actor)

    elif simulation_type in [ids.SIM_CONVEX_GAIN, ids.SIM_CONVEX_GAIN_REV_RESET, ids.SIM_CONVEX_FAKE_OFC]:
        pg_actor = network.PGActor(input_size=tot_nr_neurons, learning_rate=float(params[4]))
        q_actor = network.QActor(input_size=tot_nr_neurons, learning_rate=float(params[5]))
        gain_modulator = network.GainModulator(representation_size=tot_nr_neurons, v_lr=float(params[6]))
        if simulation_type == ids.SIM_CONVEX_GAIN:
            agent = agents.ConvexAgentGain(habit_actor=pg_actor, goal_actor=q_actor, gain_modulator=gain_modulator)
        elif simulation_type == ids.SIM_CONVEX_GAIN_REV_RESET:
            agent = agents.ConvexAgentRevReset(habit_actor=pg_actor, goal_actor=q_actor, gain_modulator=gain_modulator)
        elif simulation_type == ids.SIM_CONVEX_FAKE_OFC:
            agent = agents.ConvexAgentFakeOFC(habit_actor=pg_actor, goal_actor=q_actor, gain_modulator=gain_modulator)
        else:
            raise NotImplementedError(simulation_type)

    elif simulation_type in [ids.SIM_CONVEX_OFC, ids.SIM_FAKE_TD_SWITCH, ids.SIM_FAKE_XAP_SWITCH,
                             ids.SIM_OFC_XAP_SWITCH, ids.SIM_CONVEX_OFC_VIP_KO]:
        ofc_multi_const = float(params[4])
        ofc_thresh = float(params[5])
        policy_lr = float(params[6])
        q_lr = float(params[7])
        td_learning_rate = float(params[8])
        actor = network.PGActor(input_size=tot_nr_neurons, learning_rate=policy_lr)

        if simulation_type in [ids.SIM_CONVEX_OFC, ids.SIM_CONVEX_OFC_VIP_KO]:
            vip_ko = simulation_type == ids.SIM_CONVEX_OFC_VIP_KO
            gain_modulator = network.GainModulatorOFCSST(
                representation_size=tot_nr_neurons, v_lr=td_learning_rate, ofc_lr=q_lr,
                multi_const=ofc_multi_const, ofc_threshold=ofc_thresh, vip_ko=vip_ko
            )
            agent = agents.ConvexAgentOFC(habit_actor=actor, gain_modulator=gain_modulator)
        elif simulation_type == ids.SIM_FAKE_TD_SWITCH:
            gain_modulator = network.ContextualGainModulator(
                representation_size=tot_nr_neurons, v_lr=td_learning_rate, ofc_lr=q_lr, multi_const=ofc_multi_const,
                ofc_threshold=ofc_thresh
            )
            agent = agents.AgentFakeContextualTD(habit_actor=actor, gain_modulator=gain_modulator)

        elif simulation_type == ids.SIM_FAKE_XAP_SWITCH:
            gain_modulator = network.GainModulatorOFCSST(
                representation_size=tot_nr_neurons, v_lr=td_learning_rate, ofc_lr=q_lr,
                multi_const=ofc_multi_const, ofc_threshold=ofc_thresh
            )
            agent = agents.AgentFakeContextualXAP(habit_actor=actor, gain_modulator=gain_modulator)

        elif simulation_type == ids.SIM_OFC_XAP_SWITCH:
            gain_modulator = network.GainModulatorOFCSST(
                representation_size=tot_nr_neurons, v_lr=td_learning_rate, ofc_lr=q_lr,
                multi_const=ofc_multi_const, ofc_threshold=ofc_thresh
            )
            agent = agents.AgentContextualXAP(habit_actor=actor, gain_modulator=gain_modulator)

        else:
            raise NotImplementedError(simulation_type)
    else:
        raise NotImplementedError(simulation_type)

    # Init task environment
    task = tasks.get_task(task_id=task_id, nr_distractor=nr_noise, nr_signal=nr_signal,
                          distractor_amplitude=distractor_amplitude, signal_amplitude=signal_amplitude)

    if log_type is None:
        logger = None
    elif log_type == ids.LT_CONVEX:
        logger = [np.array([NO_VALUE] * sum(trial_nrs)) for _ in range(6)]
    elif log_type == ids.LT_OFC:
        logger = [np.array([NO_VALUE] * sum(trial_nrs)) for _ in range(5)]
    elif log_type == ids.LT_GAIN:
        logger = [np.array([NO_VALUE] * sum(trial_nrs)) for _ in range(4)]
    elif log_type == ids.LT_RGAIN:
        logger = [np.array([NO_VALUE] * sum(trial_nrs)) for _ in range(2)]
    elif log_type == ids.LT_SEL:
        logger = [np.array([NO_VALUE] * sum(trial_nrs)) for _ in range(0)]
    elif log_type == ids.LT_AGENT:
        if simulation_type in ids.TD_LR_SIMULATIONS:
            logger = [np.array([NO_VALUE] * sum(trial_nrs)) for _ in range(5)]
        else:
            logger = [np.array([NO_VALUE] * sum(trial_nrs)) for _ in range(2)]
    elif log_type == ids.LT_ALL:
        logger = [np.array([NO_VALUE] * sum(trial_nrs)) for _ in range(7)]
    else:
        raise NotImplementedError

    return agent, task, logger


def run_simulation(task_id: ids.TaskID, simulation_type: ids.SimulationType, params: List[str], seed: int = 0,
                   number_contexts: int = 2, number_trials: tuple = None, log_type: ids.LogType = None
                   ) -> Tuple[np.ndarray, Tuple]:
    # Set pseudorandom seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if number_trials is None:
        number_trials = [constants.NR_TRIALS] * number_contexts
    corrects = np.array([NO_VALUE] * sum(number_trials))

    agent, task, logger = init_simulation(task_id=task_id, simulation_type=simulation_type, params=params,
                                          log_type=log_type, trial_nrs=number_trials)

    for reversal_idx in range(number_contexts):

        # Set current rule of the task
        task.set_task(reversal=reversal_idx % 2 != 0)

        # Special cases
        if simulation_type in ids.FAKE_OFC_SIMULATIONS and reversal_idx != 0:
            agent.fake_ofc()
        elif simulation_type in [ids.SIM_FAKE_TD_SWITCH, ids.SIM_FAKE_XAP_SWITCH]:
            if reversal_idx == 0:
                pass
            elif reversal_idx == 1:
                agent.store_salience_landscape()
            elif reversal_idx == 2:
                agent.recall_salience_landscape()
            else:
                raise NotImplementedError(reversal_idx)

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


def simulate_params(task_id: ids.TaskID, simulation_type: ids.SimulationType, params: List[str], seed: int = 0,
                    nr_contexts: int = 2, n_trials: tuple = None) -> np.ndarray:
    correct, _ = run_simulation(task_id=task_id, simulation_type=simulation_type, params=params, seed=seed,
                                number_contexts=nr_contexts, number_trials=n_trials)

    return correct
