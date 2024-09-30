from typing import List, Literal

# Keys for environment variables
SNKey = Literal['n_distractor', 'n_signal', 'distractor_amplitude', 'signal_amplitude', 'signal_to_noise_ratio']
NR_DISTRACTOR: SNKey = 'n_distractor'
NR_SIGNAL: SNKey = 'n_signal'
DISTRACTOR_GAIN: SNKey = 'distractor_amplitude'
SIGNAL_GAIN: SNKey = 'signal_amplitude'
SNR: SNKey = 'signal_to_noise_ratio'
SENSORY_KEYS: List[SNKey] = [NR_DISTRACTOR, NR_SIGNAL]

# Keys for learning rates
TrainKey = Literal['lr_pg', 'lr_v', 'lr_q', 'lr_ap', 'lr_sd', 'lr_ofc']
LEARNING_RATE_PG: TrainKey = 'lr_pg'
LEARNING_RATE_V: TrainKey = 'lr_v'
LEARNING_RATE_Q: TrainKey = 'lr_q'
LEARNING_RATE_APICAL: TrainKey = 'lr_ap'
LEARNING_RATE_SD: TrainKey = 'lr_sd'
LEARNING_RATE_OFC: TrainKey = 'lr_ofc'

# Simulation keys
SimulationKey = Literal['simulation', 'seed']
SIMULATION: SimulationKey = 'simulation'
SEED: SimulationKey = 'seed'

# Keys for simulation outcomes that can be selected for during parameter scanning
OutcomeKey = Literal['loss', 'performance', 'expert_fraction', 'expert_time']
LOSS: OutcomeKey = 'loss'
PERFORMANCE: OutcomeKey = 'performance'
EXPERT_FRAC: OutcomeKey = 'expert_fraction'
EXPERT_TIME: OutcomeKey = 'expert_time'

# Keys for parameters determining the effect of lOFC activity onto SST interneurons
OFCParamKey = Literal['ofc_amplification_constant', 'threshold_activity', 'ofc_in_multi_constant']
SST_B: OFCParamKey = 'threshold_activity'
SST_W: OFCParamKey = 'ofc_amplification_constant'
OFC_K_IN: OFCParamKey = 'ofc_in_multi_constant'

# Keys of variables that are tracked by the logger and will be used to identify tables and columns in the database
LogKey = Literal['gain', 'policy_weight', 'ofc_activity']
AGENT_GAIN: LogKey = 'gain'
AGENT_POLICY_WEIGHT: LogKey = 'policy_weight'
OFC_ACTIVITY: LogKey = 'ofc_activity'

Key = SNKey | TrainKey | SimulationKey | OutcomeKey | OFCParamKey
