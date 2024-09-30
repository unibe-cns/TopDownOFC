from typing import List
from ofcsst.utils import ids


def job_to_sim_params(simulation_type: ids.SimulationType, params: List[str]):
    if simulation_type in [ids.SIM_PG_NO_GAIN]:
        noise_std = float(params[0])
        signal_amplitude = float(params[1])
        nr_noise = int(params[2])
        nr_signal = int(params[3])
        learning_rate = float(params[4])
        insert_values = (noise_std, signal_amplitude, nr_noise, nr_signal, learning_rate)
    elif simulation_type in [ids.SIM_CONVEX_NO_GAIN]:
        noise_std = float(params[0])
        signal_amplitude = float(params[1])
        nr_noise = int(params[2])
        nr_signal = int(params[3])
        learning_rate_pg = float(params[4])
        learning_rate_q = float(params[5])
        insert_values = (noise_std, signal_amplitude, nr_noise, nr_signal, learning_rate_pg, learning_rate_q)
    elif simulation_type in [ids.SIM_PG_GAIN]:
        noise_std = float(params[0])
        signal_amplitude = float(params[1])
        nr_noise = int(params[2])
        nr_signal = int(params[3])
        learning_rate_pg = float(params[4])
        learning_rate_td = float(params[5])
        insert_values = (noise_std, signal_amplitude, nr_noise, nr_signal, learning_rate_pg, learning_rate_td)
    elif simulation_type in [ids.SIM_CONVEX_GAIN, ids.SIM_CONVEX_GAIN_REV_RESET, ids.SIM_CONVEX_FAKE_OFC]:
        noise_std = float(params[0])
        signal_amplitude = float(params[1])
        nr_noise = int(params[2])
        nr_signal = int(params[3])
        learning_rate_pg = float(params[4])
        learning_rate_q = float(params[5])
        learning_rate_td = float(params[6])
        insert_values = (noise_std, signal_amplitude, nr_noise, nr_signal, learning_rate_pg, learning_rate_q,
                         learning_rate_td)
    elif simulation_type in [ids.SIM_CONVEX_OFC]:
        noise_std = float(params[0])
        signal_amplitude = float(params[1])
        nr_noise = int(params[2])
        nr_signal = int(params[3])
        ofc_mult_const = float(params[4])
        ofc_thresh_cons = float(params[5])
        learning_rate_pg = float(params[6])
        learning_rate_q = float(params[7])
        learning_rate_td = float(params[8])
        insert_values = (noise_std, signal_amplitude, nr_noise, nr_signal, ofc_mult_const, ofc_thresh_cons,
                         learning_rate_pg, learning_rate_q, learning_rate_td)
    else:
        raise NotImplementedError(simulation_type)

    return insert_values
