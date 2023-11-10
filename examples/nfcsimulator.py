import numpy as np
import matplotlib.pyplot as plt
import mmwranging


if __name__ == '__main__':
    # Load aperture illumination functions (E-field) from CST simulations
    # freqs = np.array([126, 133, 140, 147, 154, 161, 168, 175, 182]) * 1E9  # Vector with simulation frequencies (Hz)
    freqs = np.array([126, 154, 182]) * 1E9  # Vector with simulation frequencies (Hz)
    z_offset = 20E-3  # z offset to reference plane (m)
    filenames = [
        './aifsim_data/E,f=%d,z=%d.txt' % (int(freq / 1E9), int(z_offset * 1E3))
        for freq in freqs
    ]

    # Set parameters
    c = 299792458
    w_c = 2 * np.pi * np.mean(freqs)  # Center angular frequency (1/s)
    wavelength = 2 * np.pi * c / w_c  # (m)
    antenna_gain = 34.4  # (dBi)
    target_diameter = 20E-3  # (m)
    target_rcs = 10 * np.log10(np.pi**3 * target_diameter**4 / (4 * wavelength**2))

    # Initialize PPVSimulator
    sim = mmwranging.PPVSimulator(antenna_gain=antenna_gain, target_rcs=target_rcs)
    sim.load_aif_from_cst_export(freqs, filenames, 'circular', 36E-3)
    sim.init_target_meshgrid('circular', target_diameter)

    # Do simulation
    r = np.linspace(0.1, 6, 100)  # Vector with reference plane-to-target distances
    print('Start simulation!')
    pulse_position_variation, pulse_phase_variation, pulse_amplitude_gain = np.empty([3, r.size])
    for idx, r_ in enumerate(r):
        sim.set_target_position([0, 0, r_])
        sim.start_simulation()
        pulse_position_variation[idx] = sim.pulse_position_variation
        pulse_phase_variation[idx] = sim.pulse_phase_variation
        pulse_amplitude_gain[idx] = sim.pulse_amplitude_gain
        print('%d / %d done!' % (idx + 1, r.size))

    # Initialize plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8 * 1, 2.5 * 2))
    fig.suptitle('Simulation Results')
    # Plot distance variations
    ax1.plot(r, pulse_position_variation * c / 2 * 1E6, label='Pulse position')
    ax1.plot(r, -pulse_phase_variation * c / (2 * w_c) * 1E6, label='Pulse phase')
    ax1.set_xlabel('Actual Distance (m)')
    ax1.set_ylabel('Distance Variation (µm)')
    ax1.legend()
    # Plot amplitude gain factor
    ax2.plot(r, pulse_amplitude_gain)
    ax2.set_xlabel('Actual Distance (m)')
    ax2.set_ylabel('Amplitude Gain Factor (dB)')

    # Save data
    if False:
        np.savez('./nfcsim_data/nfcsim_%dmm_z=%dmm.npz' % (int(target_diameter * 1E3), int(z_offset * 1E3)),
                 r=r,
                 pulse_position_variation=pulse_position_variation,
                 pulse_phase_variation=pulse_phase_variation)

    # Start event handler
    plt.tight_layout()
    plt.show()
