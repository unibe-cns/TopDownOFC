"""Full agents, composing actors with a gain modulator and the lOFC circuit.

An agent owns the trial loop's three operations: `get_action` maps the task's
representation to a decision, `update` applies the reward to every learning
component, and `log` records selected internal variables into the caller's
arrays.

The hierarchy runs from `AgentNoGain` (a bare actor) through `AgentGain` (actor
plus gain modulator) to the convex agents, which arbitrate between a habitual
policy-gradient actor and a goal-directed Q actor with a weight set by the
agent's confidence. `ConvexAgentOFC` is the main model of the manuscript: its
arbitration and its apical gain are both driven by the lOFC. `AgentContextualXAP`
adds the context switch, which happens when the context prediction error
crosses a threshold value.
"""
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from random import random
from ofcsst.simulation import network
from ofcsst.utils import constants, ids


class Agent(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def get_representation(self, basal_input: torch.Tensor) -> torch.Tensor:
        return basal_input

    @abstractmethod
    def get_action(self, basal_input: torch.Tensor) -> bool:
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def update(self, reward: torch.Tensor):
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def log(self, logger: List, i: int, log_type: ids.LogType):
        raise NotImplementedError("Abstract method")


class AgentNoGain(Agent):
    """A bare actor with no gain modulation: the no-top-down control model."""
    def __init__(self, actor: network.Actor):
        super().__init__()
        self.actor = actor

    def get_action(self, basal_input: torch.Tensor) -> bool:
        return self.actor.simulate(representation=basal_input)

    def update(self, reward: torch.Tensor):
        self.actor.update(reward=reward)

    def log(self, logger: List, i: int, log_type: ids.LogType):
        raise NotImplementedError


class AgentGain(Agent):
    """An actor whose sensory input is scaled by a gain modulator.

    The modulator sees the already-gained representation, so the gain it learns
    compounds across trials.
    """
    def __init__(self, actor: network.ActorType, gain_modulator: network.GainModulator):
        super().__init__()
        self._actor = actor
        self._gain_modulator = gain_modulator

    def get_representation(self, basal_input: torch.Tensor) -> torch.Tensor:
        return basal_input * self._gain_modulator.get()

    def get_action(self, basal_input: torch.Tensor) -> bool:
        representation = self.get_representation(basal_input=basal_input)
        action = self._actor.simulate(representation=representation)
        self._gain_modulator.simulate(representation=representation)

        return action

    def update(self, reward: torch.Tensor):
        self._actor.update(reward=reward)
        self._gain_modulator.update(reward=reward)

    def log(self, logger: List, i: int, log_type: ids.LogType, *args, **kwargs):
        if log_type == ids.LT_GAIN:
            assert isinstance(self._gain_modulator, network.GainModulatorV)
            gains = self._gain_modulator.get()
            logger[0][i] = gains[0].item()
            logger[1][i] = gains[1].item()
            logger[2][i] = torch.mean(gains[2:]).item()
            logger[3][i] = torch.mean(
                torch.abs(self._gain_modulator._v_predictor.get_apical_credit()[2:])
            ).item()
        elif log_type == ids.LT_RGAIN:
            gains = self._gain_modulator.get()
            logger[0][i] = gains[0].item() / torch.mean(gains[1:]).item()
            logger[1][i] = (
                gains[1].item() * (len(gains) - 1) / (torch.sum(gains) - gains[1])
            )
        else:
            raise NotImplementedError(log_type)
        return logger


class ConvexAgent(Agent):
    """Arbitration between a habitual and a goal-directed actor.

    Both actors propose an action. When they disagree, the habitual one wins with
    probability `_p_habit`, which grows as the estimated outcome variance falls,
    so a confident agent becomes progressively more habitual.
    """
    def __init__(
        self,
        goal_actor: network.QActor,
        habit_actor: network.PGActor,
        sd_lr: float = constants.SD_LR,
    ):
        super().__init__()
        self._goal_actor = goal_actor
        self._habit_actor = habit_actor
        self._sd_predictor = network.VarianceEstimator(
            learning_rate=sd_lr, max_sd=constants.MAX_SD
        )
        self._p_habit = torch.Tensor([0.0])

    def get_action(self, basal_input: torch.Tensor) -> bool:
        goal_suggestion = self._goal_actor.simulate(representation=basal_input)
        habit_suggestion = self._habit_actor.simulate(representation=basal_input)
        if goal_suggestion == habit_suggestion:
            action = goal_suggestion
        else:
            if random() < self._p_habit:
                action = habit_suggestion
                self._goal_actor.set_action(action=action)
            else:
                action = goal_suggestion
                self._habit_actor.set_other_action()

        return action

    def get_r_pred(self, action: int | None = None):
        if action is None:
            return self._goal_actor.get_prediction().item()
        else:
            return self._goal_actor.get_prediction(action_id=action).item()

    def update(self, reward: torch.Tensor):
        self._goal_actor.update(reward=reward)
        self._habit_actor.update(reward=reward)
        spe = torch.abs(reward - self._goal_actor.get_prediction())
        self._sd_predictor.update(prediction_error=spe)
        self._p_habit = (
            torch.relu(1.0 - self._sd_predictor.get() / constants.MAX_SD) ** 2
        )

    def log(self, logger: List, i: int, log_type: ids.LogType):
        if log_type == ids.LT_CONVEX:
            logger[0][i] = self._p_habit
            logger[1][i] = self._sd_predictor.get()
            weights = self._habit_actor.get_weights()[0, :]
            logger[2][i] = (weights[0]).item()
            logger[3][i] = (weights[1]).item()
            logger[4][i] = torch.sum(weights[2:]).item()
            logger[5][i] = self._goal_actor.get_prediction(action_id=0)
        elif log_type == ids.LT_AGENT:
            weights = self._habit_actor.get_weights()[0, :]
            logger[0][i] = (weights[0]).item()
            logger[1][i] = (weights[1]).item()
        else:
            raise NotImplementedError
        return logger


class ConvexAgentGain(ConvexAgent):
    """A convex agent whose sensory input is scaled by a gain modulator."""
    def __init__(
        self,
        goal_actor: network.QActor,
        habit_actor: network.PGActor,
        gain_modulator: network.GainModulator,
    ):
        super().__init__(goal_actor=goal_actor, habit_actor=habit_actor)
        self._gain_modulator = gain_modulator

    def get_representation(self, basal_input: torch.Tensor) -> torch.Tensor:
        return basal_input * self._gain_modulator.get()

    def get_action(self, basal_input: torch.Tensor) -> bool:
        representation = self.get_representation(basal_input=basal_input)
        action = super().get_action(basal_input=representation)
        self._gain_modulator.simulate(representation=representation)
        return action

    def update(self, reward: torch.Tensor):
        super().update(reward=reward)
        self._gain_modulator.update(reward=reward)


class ConvexAgentGainV(ConvexAgentGain):
    """A convex gain agent specialised to the apical (value-driven) modulator.

    Narrowing the modulator type is what makes the gain and apical-credit log
    types available.
    """
    def __init__(
        self,
        goal_actor: network.QActor,
        habit_actor: network.PGActor,
        gain_modulator: network.GainModulatorV,
    ):
        super(ConvexAgentGain, self).__init__(
            goal_actor=goal_actor, habit_actor=habit_actor
        )
        self._gain_modulator = gain_modulator

    def log(self, logger: List, i: int, log_type: ids.LogType = ids.LT_AGENT):
        if log_type == ids.LT_GAIN:
            gains = self._gain_modulator.get()
            logger[0][i] = gains[0].item()
            logger[1][i] = gains[1].item()
            logger[2][i] = torch.mean(gains[2:]).item()
            logger[3][i] = torch.mean(
                torch.abs(self._gain_modulator._v_predictor.get_apical_credit()[2:])
            ).item()
        elif log_type == ids.LT_AGENT:
            gains = self._gain_modulator.get()
            ap_in_distractor = torch.mean(gains[2:])
            logger[0][i] = (torch.mean(gains[0]) / ap_in_distractor).item()
            logger[1][i] = (torch.mean(gains[1]) / ap_in_distractor).item()
            weights = self._habit_actor.get_weights()[0, :]
            logger[2][i] = (weights[0]).item()
            logger[3][i] = (weights[1]).item()
            logger[4][i] = torch.sum(weights[2:]).item() / len(weights[2:])
        elif log_type == ids.LT_ALL:
            assert isinstance(self._gain_modulator, network.GainModulatorV)
            gains = self._gain_modulator.get()
            logger[0][i] = gains[0].item()
            logger[1][i] = gains[1].item()
            logger[2][i] = torch.mean(gains[2:]).item()
            logger[3][i] = torch.mean(
                torch.abs(self._gain_modulator._v_predictor.get_apical_credit()[2:])
            ).item()
            weights = self._habit_actor.get_weights()[0, :]
            logger[4][i] = (weights[0]).item()
            logger[5][i] = (weights[1]).item()
            logger[6][i] = torch.sum(weights[2:]).item()
        else:
            logger = super().log(logger=logger, i=i, log_type=log_type)
        return logger


class ConvexAgentRevReset(ConvexAgentGainV):
    """Control model: the apical landscape is reset outright at a rule switch.

    Stands in for a maximally effective lOFC signal, delivered by the experimenter
    rather than inferred from the context prediction error.
    """
    def __init__(
        self,
        goal_actor: network.QActor,
        habit_actor: network.PGActor,
        gain_modulator: network.GainModulatorV,
    ):
        super().__init__(
            goal_actor=goal_actor,
            habit_actor=habit_actor,
            gain_modulator=gain_modulator,
        )

    def fake_ofc(self):
        """Emulate the lOFC reset of the apical landscape at a rule switch."""
        self._gain_modulator.reset()


class ConvexAgentFakeOFC(ConvexAgentGainV):
    """Control model: a fixed 100-trial SST burst is imposed at a rule switch.

    Like ConvexAgentRevReset the switch is handed to the agent, but here it acts
    through the SST pathway rather than clearing the apical weights.
    """
    def __init__(
        self,
        goal_actor: network.QActor,
        habit_actor: network.PGActor,
        gain_modulator: network.GainModulatorV,
    ):
        super().__init__(
            goal_actor=goal_actor,
            habit_actor=habit_actor,
            gain_modulator=gain_modulator,
        )
        self._ofc_active: int = 0
        self._sst_inactive: float = self._gain_modulator.get_sst()
        self._sst_active: float = 10.0

    def update(self, reward: torch.Tensor):
        super().update(reward=reward)
        if self._ofc_active:
            self._ofc_active = self._ofc_active - 1
            if self._ofc_active == 0:
                self._gain_modulator.set_sst(sst=self._sst_inactive)

    def fake_ofc(self):
        """Emulate a transient lOFC-driven SST burst lasting 100 trials."""
        self._ofc_active = 100
        self._gain_modulator.set_sst(self._sst_active)


class ConvexAgentOFC(Agent):
    """The model of the manuscript: arbitration and apical gain both driven by lOFC.

    The goal-directed proposal is read off the lOFC's own outcome predictors via a
    softmax, and the habitual actor wins a disagreement with probability equal to
    the lOFC's confidence. The same lOFC drives SST interneurons, which suppress
    the apical gain when the context prediction error exceeds threshold.
    """
    def __init__(
        self,
        habit_actor: network.PGActor,
        gain_modulator: network.GainModulatorOFCSST,
        temperature: float = 10.0,
    ):
        super().__init__()
        self._habit_actor = habit_actor
        self._gain_modulator = gain_modulator
        self._temperature = temperature

    def get_representation(self, basal_input: torch.Tensor) -> torch.Tensor:
        return basal_input * self._gain_modulator.get()

    def get_action(self, basal_input: torch.Tensor) -> bool:
        representation = self.get_representation(basal_input=basal_input)
        self._gain_modulator.simulate(representation=representation)
        qpreds = self._gain_modulator.get_q_preds()
        activations = [torch.exp(qpreds[i] / self._temperature) for i in range(2)]
        action_prob = (activations[1] / (activations[0] + activations[1])).item()
        goal_suggestion = bool(
            np.random.choice(a=[0, 1], p=[1 - action_prob, action_prob])
        )
        habit_suggestion = self._habit_actor.simulate(representation=representation)
        if goal_suggestion == habit_suggestion:
            action = goal_suggestion
        else:
            if random() < self._gain_modulator.get_confidence():
                action = habit_suggestion
            else:
                action = goal_suggestion
                self._habit_actor.set_other_action()
        self._gain_modulator.set_action(action=int(action))

        return action

    def update(self, reward: torch.Tensor):
        self._habit_actor.update(reward=reward)
        self._gain_modulator.update(reward=reward)

    def log(self, logger: List, i: int, log_type: ids.LogType = ids.LT_OFC):
        if log_type == ids.LT_OFC:
            gains = self._gain_modulator.get()
            ap_in_distractor = torch.mean(gains[2:])
            logger[0][i] = (torch.mean(gains[0]) / ap_in_distractor).item()
            logger[1][i] = (torch.mean(gains[1]) / ap_in_distractor).item()
            weights = self._habit_actor.get_weights()[0, :]
            logger[2][i] = (weights[0]).item()
            logger[3][i] = (weights[1]).item()
            logger[4][i] = self._gain_modulator.get_ofc().item()
        elif log_type == ids.LT_GAIN:
            gains = self._gain_modulator.get()
            logger[0][i] = gains[0].item()
            logger[1][i] = gains[1].item()
            logger[2][i] = torch.mean(gains[2:]).item()
            logger[3][i] = torch.mean(
                torch.abs(self._gain_modulator._v_predictor.get_apical_credit()[2:])
            ).item()
        elif log_type == ids.LT_AGENT:
            gains = self._gain_modulator.get()
            ap_in_distractor = torch.mean(gains[2:])
            logger[0][i] = (torch.mean(gains[0]) / ap_in_distractor).item()
            logger[1][i] = (torch.mean(gains[1]) / ap_in_distractor).item()
            weights = self._habit_actor.get_weights()[0, :]
            logger[2][i] = (weights[0]).item()
            logger[3][i] = (weights[1]).item()
            logger[4][i] = torch.sum(weights[2:]).item() / len(weights[2:])
        else:
            raise NotImplementedError(log_type)
        return logger


class AgentContextualXAP(ConvexAgentOFC):
    """The multi-context model: the lOFC itself triggers the context switch.

    The agent swaps its current apical landscape for the stored one when the
    lOFC's SST drive crosses `change_threshold`, but only once confidence has
    first risen past CONFIDENCE_SWITCH. That arming step stops the noisy, low
    confidence period early in learning from triggering spurious switches.
    """
    def __init__(
        self,
        habit_actor: network.PGActor,
        gain_modulator: network.GainModulatorOFCSST,
        change_threshold: float = constants.CPE_SWITCH,
    ):
        super().__init__(habit_actor=habit_actor, gain_modulator=gain_modulator)
        self._wap_stored = torch.zeros(self._gain_modulator.get_rep_size())
        self._change_context_threshold = change_threshold
        self._counter = 0
        self._switch_ready = False

    def update(self, reward: torch.Tensor):
        super().update(reward=reward)
        if (
            self._switch_ready
            and self._gain_modulator.get_ofc_sst() > self._change_context_threshold
        ):
            temp = self._gain_modulator.get_ap()
            self._gain_modulator.set_ap(ap=self._wap_stored)
            self._wap_stored = temp
            self._gain_modulator.reset_ofc()
            self._switch_ready = False
        elif (
            not self._switch_ready
            and self._gain_modulator.get_confidence() > constants.CONFIDENCE_SWITCH
        ):
            self._switch_ready = True
