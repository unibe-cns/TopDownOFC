import copy
import random
import torch
import numpy as np
from abc import ABC, abstractmethod
from ofcsst.utils import constants, ids


class VarianceEstimator:
    def __init__(self, learning_rate: float = constants.SD_LR, max_sd: float = constants.MAX_SD):
        self._predictor = torch.nn.parameter.Parameter(data=torch.tensor([max_sd]))
        self._lr = learning_rate
        self._baseline = max_sd

    def get(self):
        return self._predictor.clone().detach()

    def update(self, prediction_error: torch.Tensor):
        self._predictor = (1 - self._lr) * self._predictor + self._lr * prediction_error ** 2

    def reset(self):
        self._predictor = torch.nn.parameter.Parameter(data=torch.tensor([self._baseline]))


class SurprisePredictor:

    def __init__(self, learning_rate: float = constants.OFC_LR, baseline_value: float = 0.):
        self._predictor = torch.nn.parameter.Parameter(data=torch.tensor([baseline_value]))
        self._optimizer = torch.optim.SGD(params=[self._predictor], lr=learning_rate)
        self._loss_function = torch.nn.MSELoss()

    def get(self) -> torch.Tensor:
        return self._predictor.clone().detach()

    def update(self, surprise: torch.Tensor):
        loss = self._loss_function(surprise, self._predictor)
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

    def reset(self, baseline_value: float = 0.):
        self._predictor.data = torch.tensor([baseline_value])


class LinearPredictor(torch.nn.Linear):
    def __init__(self, input_size: int, learning_rate: float, out_size: int = 1):
        super().__init__(in_features=input_size, out_features=out_size, bias=False)
        torch.nn.init.normal_(self.weight.data, mean=0., std=1/input_size)
        self._prediction = self.forward(input=torch.zeros(input_size))
        self._optimizer = torch.optim.SGD(params=self.parameters(), lr=learning_rate)
        self._loss_function = torch.nn.MSELoss()

    def simulate(self, representation: torch.Tensor):
        self._prediction = self.forward(input=representation)

    def get_prediction(self) -> torch.Tensor:
        return self._prediction.clone().detach()

    def update(self, target: torch.Tensor, l1k: float = 0.):
        loss = self._loss_function(target, self._prediction) + l1k * torch.sum(torch.abs(self.weight))
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()


class OutcomePredictor(LinearPredictor):
    def __init__(self, input_size: int, learning_rate: float):
        super().__init__(input_size=input_size, learning_rate=learning_rate, out_size=2)
        torch.nn.init.normal_(self.weight.data, mean=0., std=1/input_size)
        self._outcome_prediction = torch.sigmoid(self.forward(input=torch.zeros(input_size)))
        self._r_prediction = torch.sigmoid(self.forward(input=torch.zeros(input_size)))
        self.valence = torch.tensor([1., -1])

    def simulate(self, representation: torch.Tensor):
        self._outcome_prediction = torch.sigmoid(self.forward(input=representation))
        # self._outcome_prediction = self.forward(input=representation)
        self._r_prediction = torch.dot(self.valence, self._outcome_prediction)

    def get_prediction(self) -> torch.Tensor:
        return self._r_prediction.clone().detach()

    def update(self, target: torch.Tensor, l1k: float = 0.):
        if target > 0:
            t = torch.tensor([target.data, 0.])
        elif target < 0:
            t = torch.tensor([0., -target.data])
        else:
            t = torch.tensor([0., 0.])
        loss = self._loss_function(t, self._outcome_prediction) + l1k * torch.sum(torch.abs(self.weight))
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()


class VPredictor(LinearPredictor):

    def __init__(self, input_size: int, learning_rate: float):
        super().__init__(input_size, learning_rate)
        self._l1k = 1 / input_size

    def get_apical_credit(self) -> torch.Tensor:
        return self.weight.data.clone().detach().squeeze()

    def update(self, target, l1k: float = None):
        super().update(target=target, l1k=self._l1k)


class OFC:

    def __init__(self, input_size: int, q_learning_rate: float, sd_max: float = constants.MAX_SD,
                 surprise_lr: float = constants.OFC_LR, sd_lr: float = constants.SD_LR, vip_ko: bool = False):
        self._action_idx = -1
        self._q_predictors = [OutcomePredictor(input_size=input_size, learning_rate=q_learning_rate),
                              OutcomePredictor(input_size=input_size, learning_rate=q_learning_rate)]
        self._sd_predictor = VarianceEstimator(max_sd=sd_max, learning_rate=sd_lr)
        self._surprise_predictor = SurprisePredictor(learning_rate=surprise_lr)
        self._surprise = torch.Tensor([0.])
        self._confidence = torch.Tensor([0.])
        self._with_vip = not vip_ko

    def simulate(self, representation: torch.Tensor, action: bool = None):
        if action is None:
            action_idxs = [0, 1]
        else:
            if action:
                self._action_idx = 0
            else:
                self._action_idx = 1
            action_idxs = [self._action_idx]
        for a in action_idxs:
            self._q_predictors[a].simulate(representation=representation)

    def get(self):
        return self._surprise_predictor.get()

    def get_q(self):
        return self._q_predictors[self._action_idx].get_prediction()

    def get_log(self):
        return self._sd_predictor.get().item(), self._confidence.item(), self._surprise.item(), self.get().item()

    def get_uncertainty(self) -> torch.Tensor:
        return self._sd_predictor.get()

    def get_confidence(self) -> torch.Tensor:
        return self._confidence

    def get_surprise(self) -> torch.Tensor:
        return self._surprise.clone().detach()

    def get_q_preds(self):
        return self._q_predictors[0].get_prediction(), self._q_predictors[1].get_prediction()

    def reset(self):
        self._surprise_predictor.reset()

    def reset_uncertainty(self):
        self._sd_predictor.reset()

    def set_action(self, action_idx: int):
        self._action_idx = action_idx

    def update(self, outcome: torch.Tensor):
        spe = torch.abs(outcome - self._q_predictors[self._action_idx].get_prediction())
        self._q_predictors[self._action_idx].update(target=outcome)
        self._confidence = torch.relu(1. - self._sd_predictor.get() / constants.MAX_SD) ** 2
        self._surprise = torch.relu(spe ** 2 - self._sd_predictor.get()) * torch.sqrt(self._confidence)
        self._sd_predictor.update(prediction_error=spe)
        if self._with_vip:
            self._surprise_predictor.update(surprise=self._surprise)
        else:
            self._surprise_predictor.update(surprise=torch.Tensor([0.]))


class GainModulator:

    def __init__(self, representation_size: int, v_lr: float, apical_lr=constants.APICAL_LR):
        self._representation_size = representation_size
        self._v_lr = v_lr
        self._gains = torch.ones(representation_size)
        self._max = torch.nn.Threshold(threshold=0., value=0.)
        self._v_predictor = VPredictor(input_size=representation_size, learning_rate=v_lr)
        self._apical_inputs = torch.zeros(representation_size)
        self._apical_lr = apical_lr
        self._sst0 = constants.SST0

    def get(self):
        return 1 + torch.relu(self._apical_inputs - self.get_sst())

    def get_sst(self):
        return self._sst0

    def set_sst(self, sst: float):
        self._sst0 = sst

    def get_rep_size(self):
        return self._representation_size

    def get_log(self):
        return 0, 0, 0, 0

    def get_q_prediction(self):
        return 0.

    def get_v_prediction(self):
        return self._v_predictor.get_prediction()

    def get_apical_inputs(self):
        return self._apical_inputs.clone()

    def simulate(self, representation: torch.Tensor):
        self._v_predictor.simulate(representation=torch.nn.functional.normalize(representation, p=1, dim=0))
        apical_ltp = self._max(self._v_predictor.get_apical_credit()) * representation
        self._apical_inputs += self._apical_lr * (apical_ltp - self.get_sst())
        self._apical_inputs = torch.clamp(input=self._apical_inputs, min=0., max=constants.MAX_GAIN-1)

    def update(self, reward: torch.Tensor):
        self._v_predictor.update(target=reward)

    def reset(self):
        self._apical_inputs = torch.zeros(self._representation_size)
        self._v_predictor = VPredictor(input_size=self._representation_size, learning_rate=self._v_lr)


class GainModulatorOFCSST(GainModulator):
    def __init__(self, representation_size: int, v_lr: float, multi_const: float, ofc_threshold: float,
                 ofc_lr: float = None, vip_ko: bool = False):
        super().__init__(representation_size=representation_size, v_lr=v_lr)
        if ofc_lr is None:
            ofc_lr = v_lr
        self._ofc = OFC(input_size=representation_size, q_learning_rate=ofc_lr, vip_ko=vip_ko)
        self._sst_multi_const = multi_const
        self._sst_threshold = ofc_threshold

    def get_q_prediction(self):
        return self._ofc.get_q()

    def get_q_preds(self):
        return self._ofc.get_q_preds()

    def get_uncertainty(self):
        return self._ofc.get_uncertainty()

    def get_surprise(self):
        return self._ofc.get_surprise()

    def get_ofc_sst(self):
        return self._sst_multi_const * self._max(self._ofc.get() - self._sst_threshold)

    def get_confidence(self):
        return self._ofc.get_confidence()

    def get_sst(self):
        return super().get_sst() + self.get_ofc_sst()

    def get_ofc(self):
        return self._ofc.get()

    def reset_ofc(self):
        self._ofc.reset()

    def set_action(self, action: int):
        self._ofc.set_action(action_idx=action)

    def simulate(self, representation: torch.Tensor):
        super().simulate(representation=representation)
        self._ofc.simulate(representation=representation)

    def update(self, reward: torch.Tensor):
        self._v_predictor.update(target=reward)
        self._ofc.update(outcome=reward)

    def get_ap(self):
        return self._apical_inputs.clone()

    def set_ap(self, ap: torch.Tensor):
        self._apical_inputs = ap


class ContextualGainModulator(GainModulatorOFCSST):
    def __init__(self, representation_size: int, v_lr: float, ofc_lr: float, multi_const: float, ofc_threshold: float):
        super().__init__(representation_size=representation_size, v_lr=v_lr, ofc_lr=ofc_lr, multi_const=multi_const,
                         ofc_threshold=ofc_threshold)
        self._stored_apical_inputs = torch.zeros(0)

    def store_apical_inputs(self):
        self._stored_apical_inputs = self._apical_inputs.clone()

    def recall_apical_inputs(self):
        self._apical_inputs = self._stored_apical_inputs.clone()

    def copy_into(self, gm: GainModulatorOFCSST, q_lr: float):
        self._ofc._sd_predictor._predictor = copy.deepcopy(gm._ofc._sd_predictor._predictor.detach())
        self._ofc._surprise_predictor = copy.deepcopy(gm._ofc._surprise_predictor)
        for a in range(2):
            self._ofc._q_predictors[a].weight = torch.nn.parameter.Parameter(copy.deepcopy(gm._ofc._q_predictors[a].weight.detach()))
            self._ofc._q_predictors[a]._optimizer = torch.optim.SGD([self._ofc._q_predictors[a].weight], q_lr)
        self._apical_inputs = copy.deepcopy(gm._apical_inputs)

    def store_td(self):
        return self._apical_inputs.clone(), copy.deepcopy(self._v_predictor.weight.data)

    def restore_td(self, stored_td):
        self._apical_inputs = stored_td[0]
        self._v_predictor.weight.data = stored_td[1]


class Actor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, *args, **kwargs) -> bool:
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError("Abstract method")


class NaiveActor(Actor):
    def __init__(self):
        super().__init__()

    def simulate(self, representation=None) -> bool:
        return bool(random.getrandbits(1))

    def update(self) -> None:
        pass


class PGActor(Actor):
    def __init__(self, input_size: int, learning_rate: float, habit: bool = True,
                 optim: ids.OptimizerType = ids.OP_SGD):
        super().__init__()
        self._policy = torch.nn.Linear(in_features=input_size, out_features=1)
        if habit:
            self._policy.bias.data.fill_(-np.log(99))
            self._w_min = 0.
        else:
            self._w_min = -constants.POLICY_WEIGHT_CLAMP
        self._w_max = constants.POLICY_WEIGHT_CLAMP
        if optim == ids.OP_SGD:
            self._optimizer = torch.optim.SGD(params=[self._policy.weight], lr=learning_rate)
        elif optim == ids.OP_ADAM:
            self._optimizer = torch.optim.Adam(params=[self._policy.weight], lr=learning_rate)
        elif optim == ids.OP_ADAGRAD:
            self._optimizer = torch.optim.Adagrad(params=[self._policy.weight], lr=learning_rate)
        elif optim == ids.OP_ADADELTA:
            self._optimizer = torch.optim.Adadelta(params=[self._policy.weight], lr=learning_rate)
        elif optim == ids.OP_ADAMAX:
            self._optimizer = torch.optim.Adamax(params=[self._policy.weight], lr=learning_rate)
        elif optim == ids.OP_RMSP:
            self._optimizer = torch.optim.RMSprop(params=[self._policy.weight], lr=learning_rate)
        else:
            raise NotImplementedError(optim)
        self._action_prob = self.get_action_prob(representation=torch.zeros((input_size,)))
        self._lick = False

    def get_weights(self):
        return self._policy.weight.clone().detach()

    def get_action_prob(self, representation: torch.Tensor) -> torch.Tensor:
        activation = self._policy.forward(input=representation)
        return torch.sigmoid(input=activation) * 0.98 + 0.01

    def simulate(self, representation: torch.Tensor) -> bool:
        action_lick_prob = self.get_action_prob(representation=representation)
        l_prob = action_lick_prob.clone().detach().item()
        self._lick = bool(np.random.choice([0, 1], p=[1. - l_prob, l_prob]))
        if self._lick:
            self._action_prob = action_lick_prob
        else:
            self._action_prob = 1 - action_lick_prob

        return self._lick

    def set_other_action(self):
        self._action_prob = 1 - self._action_prob

    def update(self, reward: torch.Tensor, *args, **kwargs) -> None:
        loss = -torch.mul(torch.log(self._action_prob), reward)
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        torch.clamp(self._policy.weight.data, min=self._w_min, max=self._w_max, out=self._policy.weight.data)


class QActor(Actor):
    def __init__(self, input_size: int, learning_rate: float):
        super().__init__()
        self._q_predictors = [OutcomePredictor(input_size=input_size, learning_rate=learning_rate),
                              OutcomePredictor(input_size=input_size, learning_rate=learning_rate)]
        self.temperature = constants.Q_TEMPERATURE
        self.q_idx = 0
        self._input_size = input_size
        self._learning_rate = learning_rate

    def get_prediction(self, action_id: int = None):
        if action_id is None:
            return self._q_predictors[self.q_idx].get_prediction()
        else:
            return self._q_predictors[action_id].get_prediction()

    def simulate(self, representation: torch.Tensor) -> bool:
        for i in range(2):
            self._q_predictors[i].simulate(representation=representation)
        activations = [torch.exp(self._q_predictors[i].get_prediction() / self.temperature) for i in range(2)]
        action_prob = (activations[0] / (activations[0] + activations[1])).item()
        action = bool(np.random.choice([0, 1], p=[1. - action_prob, action_prob]))
        self.set_action(action=action)
        return action

    def set_action(self, action: bool):
        if action:
            self.q_idx = 0
        else:
            self.q_idx = 1

    def update(self, reward: torch.Tensor, *args, **kwargs) -> None:
        self._q_predictors[self.q_idx].update(target=reward)

    def reset_predictors(self):
        self._q_predictors = [OutcomePredictor(input_size=self._input_size, learning_rate=self._learning_rate),
                              OutcomePredictor(input_size=self._input_size, learning_rate=self._learning_rate)]


ActorType = NaiveActor | PGActor
