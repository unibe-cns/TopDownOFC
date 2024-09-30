import random
import torch
from abc import abstractmethod, ABC
from ofcsst.utils import constants, ids


class Task2vN(ABC):
    def __init__(self, nr_distractor: int, nr_signal: int = 2):
        super().__init__()
        self.go_trial = None
        self.reversal = False
        self.reward = torch.Tensor([constants.REWARD])
        self.punishment = torch.Tensor([constants.PUNISHMENT])
        self.not_acted_outcome = torch.Tensor([0.])
        self.representation_size = nr_distractor + nr_signal
        signal_idxs = torch.tensor(list(range(nr_signal)))
        self.stimulus1_idx = signal_idxs[:int(nr_signal / 2)]
        self.stimulus2_idx = signal_idxs[int(nr_signal / 2):]
        self.correct = torch.Tensor([1])
        self.wrong = torch.Tensor([0])

    def init_trial(self, stimulus: bool = None) -> bool:
        if stimulus is None:
            stimulus = bool(random.getrandbits(1))
        self.go_trial = (stimulus and not self.reversal) or (not stimulus and self.reversal)
        return stimulus

    @abstractmethod
    def init_stimuli(self, stimulus: bool = None) -> torch.Tensor:
        raise NotImplementedError("Abstract method")

    def get_outcome(self, acted_upon) -> (torch.Tensor, torch.Tensor):
        if acted_upon:
            if self.go_trial:
                return self.reward, self.correct
            else:
                return self.punishment, self.wrong
        else:
            if self.go_trial:
                return self.not_acted_outcome, self.wrong
            else:
                return self.not_acted_outcome, self.correct

    def set_task(self, reversal: bool) -> None:
        self.reversal = reversal


class Binary2VN(Task2vN):
    def __init__(self, nr_distractor: int, nr_signal: int, distractor_amplitude: float = 1.,
                 signal_amplitude: float = 1.):
        super().__init__(nr_distractor=nr_distractor, nr_signal=nr_signal)
        self._noise_amplitude = distractor_amplitude
        self._signal_amplitude = signal_amplitude

    def init_stimuli(self, stimulus: bool = None) -> torch.Tensor:
        stimulus = self.init_trial(stimulus=stimulus)
        representation = torch.mul(torch.randint(low=0, high=2, size=(self.representation_size,)).float(),
                                   self._noise_amplitude)
        if stimulus:
            representation.index_fill_(dim=0, index=self.stimulus1_idx, value=self._signal_amplitude)
            representation.index_fill_(dim=0, index=self.stimulus2_idx, value=0.)
        else:
            representation.index_fill_(dim=0, index=self.stimulus1_idx, value=0.)
            representation.index_fill_(dim=0, index=self.stimulus2_idx, value=self._signal_amplitude)

        return representation


def get_task(task_id: ids.TaskID, nr_distractor: int, nr_signal: int, distractor_amplitude: float = 1.,
             signal_amplitude: float = 1.) -> Task2vN:
    if task_id == ids.BINARY_2VN:
        return Binary2VN(nr_distractor=nr_distractor, nr_signal=nr_signal, distractor_amplitude=distractor_amplitude,
                         signal_amplitude=signal_amplitude)
    else:
        raise NotImplementedError(task_id)
