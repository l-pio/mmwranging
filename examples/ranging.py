import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import mmwranging


if __name__ == '__main__':
    # Set diameter of radar target to load the appropriate data
    target_diameter = 20  # (mm)

    # Load measured data
    data = np.load('./ranging_data/%dmm.npz' % target_diameter)
    ifdata = data['ifdata']
    temp = data['temp']
    press = data['press']
    hum = data['hum']
    co2conc = data['co2conc']
    reference = data['reference']

    # Set model for near-field correction
    nfcsim_data = np.load('./nfcsim_data/nfcsim_%dmm_z=20mm.npz' % target_diameter)
    nfc_model = {
        None: lambda: None,
        'AM': lambda: dict(mode='AM', d1=36E-3, d2=target_diameter / 1E3, rtt_offset=500E-12),
        'PM': lambda: dict(
            mode='PM',
            a_tot={20: 12.4E-4, 30: 16.6E-4, 40: 22.2E-4, 50: 31.2E-4}[target_diameter],
            r_off={20: -10.2E-2, 30: -6.1E-2, 40: 0.8E-2, 50: 11.9E-2}[target_diameter]),
        'SM': lambda: dict(
            mode='func',
            pulse_position_variation_func=interp1d(nfcsim_data['r'] - 20E-3, nfcsim_data['pulse_position_variation']),
            pulse_phase_variation_func=interp1d(nfcsim_data['r'] - 20E-3, nfcsim_data['pulse_phase_variation']),
            rtt_offset=700E-12)
    }['PM']()  # None: no near-field correction / 'AM': approx. model / 'PM': parametric model / 'SM': simulation model

    # Initialize mmwRanging processor
    proc = mmwranging.Processor(
        154007370664,  # Center frequency (Hz)
        55983438670,  # Sweep bandwidth (Hz)
        2E-3,  # Sweep duration (sec)
        10001,  # Samples per sweep (#)
        td_length=10001 * 1,  # Length in time-domain (td_length = fd_length + zero_padding)
        use_triangular_modulation=True,  # Use triangular frequency modulation: upchirps and downchirps
        use_phase=True,  # Use pulse phase for ToF estimation
        use_sweep_interleaving=True,  # Interleave adjacent upchirps and downchirps (triangular mod.)
        if_data_order='du',  # 'du': downchirp, upchirp, downchirp, ... 'ud': upchirp, downchirp, upchirp, ...
        estimator='QIPS',  # Set parameter estimator (here: quadratic interpolation with power scaled radar echo)
        window='hann',  # Set window function
        refractive_index_model='dband',  # Set refractive-index model
        compensate_residual_phase_term=False,  # Compensate tau^2 residual phase term (unnecessary for triangular mod.)
        if_path_filter='phase',  # Compensate if-path frequency response
        nearfield_correction=nfc_model,  # Set model for near-field correction
        roi=[0.4 * 2 / 3E8, 5.65 * 2 / 3E8]  # Range-of-interest for peak search (sec)
        )

    # Load IF path filter frequency response
    ifpath_data = np.genfromtxt('./ifpath_data/if_path_sim_2piSENSE.txt', delimiter=',', skip_header=1).T
    proc.load_if_path_response(ifpath_data[0], 10**(ifpath_data[1] / 20) * np.exp(1j * ifpath_data[2] / 180 * np.pi))

    # Process data
    if proc.use_sweep_interleaving:
        dist = np.empty([ifdata.shape[0], ifdata.shape[1] - 1])
    else:
        dist = np.empty([ifdata.shape[0], ifdata.shape[1] // 2])
    snr = np.empty(ifdata.shape[0])
    td_data = np.empty(ifdata.shape * np.array([1, 1, 0]) + np.array([0, 0, proc.td_length]))

    for idx, (_ifdata, _temp, _press, _hum, _co2conc) in enumerate(zip(ifdata, temp, press, hum, co2conc)):
        # Update input data
        proc.update_if_data(_ifdata, clear_interleaving_mem=True)
        proc.update_atmospheric_data(_temp + 273.15, _press, _hum, _co2conc / 1E6)
        # Initialize the signal filter for the rf path
        if idx == 0:
            proc.do_rf_path_calibration()
        # Store output data
        dist[idx] = proc.distance
        snr[idx] = proc.snr_db
        td_data[idx] = proc.td_data_db

    time_axis = proc.time_axis

    # Compute systematic and random errors
    mean_dist = np.mean(dist, axis=-1)
    error = mean_dist + reference
    error = error - np.mean(error[mean_dist > 4])
    random_error = np.std(dist, axis=-1)

    # Plot data
    fig, axs = plt.subplots(4, 1, figsize=(8 * 1, 2.5 * 4))

    axs[0].plot(mean_dist, error * 1E6, 'o')
    axs[0].set_title('Systematic Error')
    axs[0].set_xlabel('Actual Distance (m)')
    axs[0].set_ylabel('Error (µm)')

    axs[1].plot(mean_dist, random_error * 1E9, 'o')
    axs[1].sharex(axs[0])
    axs[1].set_title('Random Error')
    axs[1].set_xlabel('Actual Distance (m)')
    axs[1].set_ylabel('Random Error (nm)')

    axs[2].plot(mean_dist, snr, 'o')
    axs[2].sharex(axs[0])
    axs[2].set_title('Signal-to-Noise Ratio')
    axs[2].set_xlabel('Actual Distance (m)')
    axs[2].set_ylabel('SNR (dB)')

    for td_data_ in td_data[:, 0]:
        axs[3].plot(time_axis[:time_axis.size // 2] * 1E9, td_data_[:time_axis.size // 2])
    axs[3].set_title('Radar Echoes')
    axs[3].set_xlabel('Time-of-Flight (nm)')
    axs[3].set_ylabel('Radar Echoes (dB)')

    # Start event handler
    plt.tight_layout()
    plt.show()
