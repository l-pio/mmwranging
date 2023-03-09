import numpy as np
from pkg_resources import resource_filename


# Load oxygen and water lines for MPM93
mpm93_o2_lines = np.loadtxt(resource_filename(__name__, './mpm93_o2_lines.dat'), skiprows=1)
mpm93_h2o_lines = np.loadtxt(resource_filename(__name__, './mpm93_h2o_lines.dat'), skiprows=1)


def saturation_vapour_pressure(temp):
    """Compute vapour pressure for a given temperature using Buck's equation.

    Parameters:
    - temp: temperature (K)

    Notes:
    - Buck 1981: "New Equations for Computing Vapor Pressure and Enhancement Factor."
    - Using equation e_w2 (0...50°C, abs. error: 0.05% at 50°C) and enhancement factor f_w1 (>0.8 bar).
    """
    es = 613.65 * np.exp((17.368 * temp - 4745) / (temp - 34.27))
    return es


def five_term_eq(freq, temp, press, hum, co2_conc, k):
    """Compute the refractivity using a five-term equation with parameters k[0...4].

    Parameters:
    - freq: frequency (Hz)
    - temp: temperature (K)
    - press: pressure (Pa)
    - hum: rel. humidity (%)
    - co2_conc: carbon dioxide concentration (0...1)
    - k: parameters of equation
    """
    es = saturation_vapour_pressure(temp)
    pw = es * hum / 100
    pc = press * co2_conc
    pd = press - pw - pc
    return k[0] * pd / temp + k[1] * pw / temp + k[2] * pw / temp ** 2 + k[3] * pc / temp + k[4] * pw / temp * freq


def smith_weintraub1953(temp, press, hum):
    """Compute the refractivity using Smith & Weintraub's equation.

    Parameters:
    - temp: temperature (K)
    - press: pressure (Pa)
    - hum: rel. humidity (%)

    Notes:
    - Smith & Weintraub 1953: "The Constants in the Equation for Atmospheric Refractive Index at Radio Frequencies."
    """
    k = [0.776, 0.72, 3750, 0, 0]
    return five_term_eq(0, temp, press, hum, 0, k)


def eq1(freq, temp, press, hum, co2_conc=None):
    """Compute the mmWave refractivity of atmosphere using Equation 1 (75...110 GHz).

    Parameters:
    - freq: frequency (Hz)
    - temp: temperature (K)
    - press: pressure (Pa)
    - hum: rel. humidity (%)
    - co2_conc: carbon dioxide concentration (0...1)

    Notes:
    - Optimized to: 0.9...1.1 bar, 0...50°C, 0...100% RH.
    """
    co2_conc = 400E-6 if co2_conc is None else co2_conc
    k = [0.7754, 0.5786, 3768, 1.335, 1.181E-12]
    return five_term_eq(freq, temp, press, hum, co2_conc, k)


def eq2(freq, temp, press, hum, co2_conc=None):
    """Compute the mmWave refractivity of atmosphere using Equation 2 (110...170 GHz).

    Parameters:
    - freq: frequency (Hz)
    - temp: temperature (K)
    - press: pressure (Pa)
    - hum: rel. humidity (%)
    - co2_conc: carbon dioxide concentration (0...1)

    Notes:
    - Optimized to: 0.9...1.1 bar, 0...50°C, 0...100% RH.
    """
    co2_conc = 400E-6 if co2_conc is None else co2_conc
    k = [0.7756, 0.3656, 3808, 1.335, 1.862E-12]
    return five_term_eq(freq, temp, press, hum, co2_conc, k)


def eq3(freq, temp, press, hum, co2_conc=None):
    """Compute the mmWave refractivity of atmosphere using Equation 3 (200...300 GHz).

    Parameters:
    - freq: frequency (Hz)
    - temp: temperature (K)
    - press: pressure (Pa)
    - hum: rel. humidity (%)
    - co2_conc: carbon dioxide concentration (0...1)

    Notes:
    - Optimized to: 0.9...1.1 bar, 0...50°C, 0...100% RH.
    """
    co2_conc = 400E-6 if co2_conc is None else co2_conc
    k = [0.7757, -0.3928, 3931, 1.335, 3.323E-12]
    return five_term_eq(freq, temp, press, hum, co2_conc, k)


def mpm93(freq, temp, press, hum):
    """Compute the complex mmWave refractivity of atmosphere using MPM93.

    Parameters:
    - freq: frequency (Hz)
    - temp: temperature (K)
    - press: pressure (Pa)
    - hum: rel. humidity (%)

    Notes:
    - Uses only the dry-air and water-vapour module of MPM93.
    - Liebe 1993: "Propagation Modeling of Moist Air and Suspended Water/Ice Particles at Frequencies Below 1000 GHz."
    - Typical range: 10E-5...1013 mbar, -100...50 °C, 0...100% RH
    """
    # Compute partial pressures
    es = saturation_vapour_pressure(temp)
    pw = es * hum / 100
    pd = press - pw
    # Simulation
    theta = 300.0 / temp  # Reciprocal temperature
    nd = mpm93_dryair_module(freq / 1E9, theta, pw / 100, pd / 100)  # Convert units to those of the MPM93 modules
    nv = mpm93_watervapour_module(freq / 1E9, theta, pw / 100, pd / 100)

    return nd + nv


def mpm93_dryair_module(ny, theta, e, pd):
    """Carry out the dry-air module of MPM93.

    Parameters:
    - ny: frequency (GHz)
    - theta: reciprocal temperature
    - e: partial pressure of water vapour (hPa)
    - pd: partial pressure of dry air (hPa)

    Notes:
    - Liebe 1993: "Propagation Modeling of Moist Air and Suspended Water/Ice Particles at Frequencies Below 1000 GHz."
    - Typical range: 10E-5...1013 mbar, -100...50 °C, 0...100% RH
    """
    # Nondispersive term (nd)
    nd = 0.2588 * pd * theta

    # Oxygen line terms (nk)
    nk = 0j
    for a in mpm93_o2_lines:
        # Line strength
        s = a[1] / a[0] * pd * theta**3 * np.exp(a[2] * (1 - theta))
        # Line width
        gamma = a[3] * 1E-3 * (pd * theta**a[4] + 1.1 * e * theta)
        # Line behaviour in the mesosphere (Zeeman-effect) -> obsolete
        # B = 60E-6  # Magnetic field strength (22...65μT)
        # gamma = np.sqrt(gamma**2 + 625 * B**2)
        # Overlap parameter of pressure-broadened lines
        delta = (a[5] + a[6] * theta) * (pd + e) * theta**0.8
        # Complex line shape function by Rosenkranz
        f = ny * ((1 - 1j * delta) / (a[0] - ny - 1j * gamma) - (1 + 1j * delta) / (a[0] + ny + 1j * gamma))
        nk += s * f

    # Nonresonant terms (nn)
    so = 6.14E-5 * pd * theta**2
    fo = -ny / (ny + 1j * 0.56E-3 * (pd + e) * theta**0.8)
    sn = 1.4E-12 * pd**2 * theta**3.5
    fn = ny / (1 + 1.9E-5 * ny**1.5)
    nn = so * fo + 1j * sn * fn

    return nd + nk + nn


def mpm93_watervapour_module(ny, theta, e, pd):
    """Carry out the water-vapour module of MPM93.

    Parameters:
    - ny: frequency (GHz)
    - theta: reciprocal temperature
    - e: partial pressure of water vapour (hPa)
    - pd: partial pressure of dry air (hPa)

    Notes:
    - Liebe 1993: "Propagation Modeling of Moist Air and Suspended Water/Ice Particles at Frequencies Below 1000 GHz."
    - Typical range: 10E-5...1013 mbar, -100...50 °C, 0...100% RH
    - Calculation of the water vapour continuum by means of a pseudo-line at 1780 GHz.
    """
    # Nondispersive term (nv)
    nv = (4.163 * theta + 0.239) * e * theta

    # H2O line spectrum (nl)
    nl = 0j
    for b in mpm93_h2o_lines:
        # Line strength
        s = b[1] / b[0] * e * theta**3.5 * np.exp(b[2] * (1 - theta))
        # Width of preasure boradened line
        gamma = b[3] / 1000 * (b[4] * e * theta**b[6] + pd * theta**b[5])
        # Doppler-broadening for pressures below 0.7 mbar
        if pd + e < 0.7:
            gamma_d = 1.46E-6 * b[0] / np.sqrt(theta)
            gamma = 0.535 * gamma + np.sqrt(0.217 * gamma**2 + gamma_d**2)
        # Complex line shape function by Rosenkranz (neglecting overlapping of lines, i.e., delta = 0)
        f = ny * (1 / (b[0] - ny - 1j * gamma) - 1 / (b[0] + ny + 1j * gamma))
        nl += s * f

    return nv + nl
