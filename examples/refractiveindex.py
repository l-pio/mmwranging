import numpy as np
import matplotlib.pyplot as plt
import mmwranging
from matplotlib.lines import Line2D


if __name__ == '__main__':
    # Init plots
    fig, axs = plt.subplots(2, 1, figsize=(8 * 1, 2.5 * 2))

    # Simulate attenuation of moist air
    freqs = np.linspace(10E9, 1000E9, 400)  # (Hz)
    temp = np.array([40, 20, 0.001]) + 273.15  # (K)
    press = 101325  # (Pa)
    rel_hum = 100  # (%)

    for _temp in temp:
        # MPM93
        refractivity = mmwranging.mpm93(freqs, _temp, press, rel_hum)
        attenuation = 0.1820 * (freqs / 1E9) * np.imag(refractivity)  # (db / km)
        axs[0].plot(freqs / 1E9, attenuation, label='%d K' % _temp)

    axs[0].set_yscale('log')
    axs[0].set_title('Attenuation of moist air at %.2f mbar, %d%%RH' % (press / 1E2, rel_hum))
    axs[0].set_xlabel('Frequency (GHz)')
    axs[0].set_ylabel('Attenuation (dB / km)')
    axs[0].legend()

    # Simulate refractive index of moist air
    freqs = np.linspace(10E9, 350E9, 400)  # (Hz)
    temp = 20 + 273.15  # (K)
    press = 101325  # (Pa)
    rel_hum = np.array([48, 50, 52])  # (%)

    for _rel_hum in rel_hum:
        # MPM93
        refractivity_mpm93 = mmwranging.mpm93(freqs, temp, press, _rel_hum)
        p = axs[1].plot(freqs / 1E9, np.real(refractivity_mpm93), ls='-', alpha=0.6)
        # Smith & Weintraub 1953
        _freqs = freqs[freqs <= 50E9]
        refractivity_sw = mmwranging.smith_weintraub1953(temp, press, _rel_hum)
        p = axs[1].plot(_freqs / 1E9, refractivity_sw * np.ones(_freqs.shape), ls='--', color=p[-1].get_color())
        # D-band equation (EQ2)
        _freqs = freqs[(freqs >= 110E9) * (freqs <= 170E9)]
        refractivity_dband = mmwranging.eq2(_freqs, temp, press, _rel_hum)
        axs[1].plot(_freqs / 1E9, refractivity_dband, ls='-.', color=p[-1].get_color())

    axs[1].set_title('Refractive index of moist air at %d K, %.2f mbar, %s%%RH' % (
        temp,
        press / 1E2,
        '/'.join('%d' % _rel_hum for _rel_hum in rel_hum)
    ))
    axs[1].set_xlabel('Frequency (GHz)')
    axs[1].set_ylabel('Refractivity (N-units)')
    axs[1].legend(handles=[
        Line2D([0], [0], label='MPM93', color='k', ls='-'),
        Line2D([0], [0], label='Smith & Weintraub 1953', color='k', ls='--'),
        Line2D([0], [0], label='D-band equation', color='k', ls='-.')])

    # Start event handler
    plt.tight_layout()
    plt.show()
