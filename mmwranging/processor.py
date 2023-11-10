import numpy as np
from scipy.signal import windows
from cached_property import cached_property
from contextlib import suppress

from .estimator import basic_estimator, qi_estimator, qips_estimator, zoom_estimator
from .refractiveindex import mpm93, smith_weintraub1953, eq1, eq2, eq3
from .nearfieldcorrection import approximate_model, approximate_model_2_phase, approximate_model_2_position,\
    gen_parametric_model


C0 = 299792458


class Processor:
    """Processor for estimating distance from measured IF-data."""
    def __init__(self, center_freq, sweep_bandwidth, sweep_duration, fd_length, **kwargs):
        """Initialize a new processor instance.

        Parameters:
        - center_freq: center frequency (Hz)
        - sweep_bandwidth: sweep bandwidth (Hz)
        - sweep duration: sweep duration (Hz)
        - fd_length: samples per sweep, i.e., in frequency domain (#)
        """
        self.center_freq = center_freq
        self.sweep_bandwidth = sweep_bandwidth
        self.sweep_duration = sweep_duration
        self.fd_length = fd_length

        self.if_data_order = kwargs.get('if_data_order', 'ud')
        self.td_length = kwargs.get('td_length', fd_length)  # Set to fd_length if no zero-padding is set
        self.window = kwargs.get('window', 'hann')
        self.if_path_filter = kwargs.get('if_path_filter', None)
        self.rf_path_filter = kwargs.get('rf_path_filter', 'IF')
        self.compensate_residual_phase_term = kwargs.get('compensate_residual_phase_term', False)
        self.time_gate = kwargs.get('time_gate', [10/self.sweep_bandwidth, 10/self.sweep_bandwidth])
        self.roi = kwargs.get('roi', None)
        self.direction = kwargs.get('direction', 'positive')
        self.estimator = kwargs.get('estimator', 'QIPS')
        self.use_triangular_modulation = kwargs.get('use_triangular_modulation', True)
        self.use_sweep_interleaving = kwargs.get('use_sweep_interleaving', True)
        self.use_phase = kwargs.get('use_phase', True)
        self.refractive_index_model = kwargs.get('refractive_index_model', None)
        self.nearfield_correction = kwargs.get('nearfield_correction', None)

        self.if_path_response = None
        self.rf_path_response = None
        self.origin = None
        self.if_data = None
        self.temp_data = None
        self.press_data = None
        self.hum_data = None
        self.co2_conc_data = None
        self.tau_mem = None
        self.phi_mem = None
        self.delta_phi = None

    def n_to_if(self, n):
        """Convert # to frequency in IF-domain."""
        return n / self.sweep_duration * (self.fd_length - 1) / self.td_length

    def n_to_f(self, n):
        """Convert # to frequency in RF-domain."""
        return (n / (self.fd_length - 1) - 1/2) * self.sweep_bandwidth + self.center_freq

    def n_to_t(self, n):
        """Convert # to time delay in time-domain."""
        return n / self.sweep_bandwidth * (self.fd_length - 1) / self.td_length

    def t_to_n(self, t):
        """Convert time delay to # in time-domain."""
        return t * self.sweep_bandwidth / (self.fd_length-1) * self.td_length

    def time_gating(self, td_data, gate):
        """Gate time-domain data within gate = [time_min, time_max]."""
        if np.ndim(td_data) == 1:
            # Single-dimensional array
            length = round(self.t_to_n(gate[1] - gate[0]) / 2) * 2  # time-gate with constant length
            center = round(self.t_to_n(np.mean(gate)))
            gate = np.zeros(td_data.shape)
            gate[center - length // 2:center + length // 2] = 1
            return td_data * gate
        else:
            # Multi-dimensional array: recursion
            return np.array([self.time_gating(_td_data, _gate) for _td_data, _gate in zip(td_data, np.transpose(gate))])

    @property
    def if_axis(self):
        """Intermediate-frequency axis for data in frequency-domain."""
        return self.n_to_if(np.arange(self.td_length))

    @property
    def time_axis(self):
        """Time axis for data in time-domain."""
        return self.n_to_t(np.arange(self.td_length))

    @property
    def freq_axis(self):
        """Frequency axis for data in frequency-domain (passband)."""
        return self.n_to_f(np.arange(self.fd_length))

    @property
    def range_axis(self):
        """Range axis for data in time-domain."""
        return self.n_to_t(np.arange(self.td_length)) / 2 * C0 / self.refractive_index

    @cached_property
    def td_data(self):
        """Time-domain data."""
        fd = np.asarray(self.if_data, complex)

        # Flip downchirp data
        if self.sl_downchirps is not None:
            fd[self.sl_downchirps] = np.flip(fd[self.sl_downchirps], axis=-1)

        # Window
        window = getattr(windows, self.window)(self.fd_length)
        fd = fd * window / np.sqrt(np.mean(window**2))  # Denormalized

        # Filter RF path response
        if self.rf_path_response is not None:
            fd = {
                None: lambda: fd,
                'IF': lambda: fd / self.rf_path_response,
                'AP': lambda: fd * np.abs(self.rf_path_response) / self.rf_path_response,
                'MF': lambda: fd * np.conj(self.rf_path_response),
                # Note: WF is not yet implemented! Window function etc. must be deactivated for gated_fd_data()
                # 'WF': lambda: (fd * np.conj(self.rf_path_response) /
                #               (np.abs(self.rf_path_response)**2 + 10**(-self.snr_fd_db / 10))),
            }[self.rf_path_filter]()

        # Centered IFFT
        td = np.fft.ifft(fd, self.td_length) * self.td_length / self.fd_length  # Normalization adjusted to zero-padding
        td = td * np.exp(-1j * np.pi * np.arange(self.td_length) * (self.fd_length - 1) / self.td_length)

        # Supress non-causal components in the pulse response (Hilbert transform)
        # for real valued if-data (non-IQ data).
        if np.isrealobj(self.if_data):
            h = np.zeros(self.td_length)
            if self.td_length % 2 == 0:
                h[0] = h[self.td_length // 2] = 1
                h[1:self.td_length // 2] = np.sqrt(2)  # +3dB
            else:
                h[0] = 1
                h[1:(self.td_length + 1) // 2] = np.sqrt(2)  # +3dB
            td = td * h

        # Compensate residual phase term (IF-domain all-pass filter)
        if self.compensate_residual_phase_term:
            h = np.exp(-1j * np.pi * self.sweep_bandwidth / self.sweep_duration * self.time_axis**2)
            if self.sl_upchirps is not None:
                td[self.sl_upchirps] *= h
            if self.sl_downchirps is not None:
                td[self.sl_downchirps] *= np.conj(h)  # Inverse phase related to upchirps

        # Filter IF-path frequency response
        if self.if_path_response is not None:
            h = {
                None: lambda: 1,
                'amp+phase': lambda: 1 / self.if_path_response,
                'amp': lambda: 1 / np.abs(self.if_path_response),
                'phase': lambda: np.abs(self.if_path_response) / self.if_path_response,
                }[self.if_path_filter]()
            if self.sl_upchirps is not None:
                td[self.sl_upchirps] *= np.conj(h)  # Note: Causal impulse response refers to negative IF frequencies
            if self.sl_downchirps is not None:
                td[self.sl_downchirps] *= h  # "

        return td

    @cached_property
    def td_data_abs(self):
        """Absolute value of time-domain data."""
        return np.abs(self.td_data)

    @cached_property
    def td_data_db(self):
        """Absolute value of time-domain data in dB."""
        return 20 * np.log10(self.td_data_abs + 1E-15)  # -300 dB floor

    @cached_property
    def td_data_rad(self):
        """Phase of time-domain data in radians."""
        return np.angle(self.td_data)

    @cached_property
    def gated_td_data(self):
        """Time-gated time-domain data."""
        # Determine time-gate regarding TOF dimension
        tof = {
            self.td_data.shape[0]: self.tof,
            self.td_data.shape[0] - 1: np.insert(self.tof, 0, self.tof[0]),
            self.td_data.shape[0] // 2: np.repeat(self.tof, 2)
        }[self.tof.shape[0]]
        gate = [tof - self.time_gate[0], tof + self.time_gate[1]]

        # Time gate
        return self.time_gating(self.td_data, gate)

    @cached_property
    def gated_td_data_abs(self):
        """Absolute value of time-gated time-domain data."""
        return np.abs(self.gated_td_data)

    @cached_property
    def gated_td_data_db(self):
        """Absolute value of time-gated time-domain data in dB."""
        return 20 * np.log10(self.gated_td_data_abs + 1E-15)  # -300 dB floor

    @cached_property
    def gated_td_data_rad(self):
        """Phase of time-gated time-domain data in radians."""
        return np.angle(self.gated_td_data)

    @cached_property
    def fd_data(self):
        """Frequency-domain data."""
        # Inversion of centered FFT
        td = self.td_data / np.exp(-1j * np.pi * np.arange(self.td_length) * (self.fd_length-1) / self.td_length)
        return np.fft.fft(td)[..., :self.fd_length] / self.td_length * self.fd_length  # Account for normalization

    @cached_property
    def gated_fd_data(self):
        """Time-gated frequency-domain data."""
        # Inversion of centered fft
        td = self.gated_td_data / np.exp(-1j * np.pi * np.arange(self.td_length) * (self.fd_length-1) / self.td_length)
        return np.fft.fft(td)[..., :self.fd_length] / self.td_length * self.fd_length  # Account for normalization

    @cached_property
    def noise(self):
        """Measured noise of data."""
        tg_length = round(self.t_to_n(np.sum(self.time_gate)) / 2) * 2
        # Process up- and downchirps separately
        noise = []
        for sl in [self.sl_upchirps, self.sl_downchirps]:
            if sl is not None:
                td = self.gated_td_data_abs[sl]
                td = (td[0:-1] - td[1:])  # Processing of adjacent radar echoes
                noise += [np.sum(td**2, axis=-1)]
        noise = np.mean(noise)
        # Account for processing gains and losses
        noise = noise * self.fd_length / tg_length / 2
        return noise

    @cached_property
    def power(self):
        """Measured power of data."""
        power = np.mean(np.sum(self.gated_td_data_abs**2, axis=-1))
        power = power * self.fd_length / self.td_length  # Account for zero-padding "gain"
        power = power - self.noise
        # Return nan if two consecutive measurements are incoherent
        return power if power > 0 else np.nan

    @cached_property
    def snr(self):
        """Measured signal-to-noise ratio of data."""
        return self.power / self.noise

    @cached_property
    def noise_db(self):
        """Measured noise of data in dB."""
        return 10 * np.log10(self.noise + 1E-30)  # -300 dB floor

    @cached_property
    def power_db(self):
        """Measured power of data in dB."""
        return 10 * np.log10(self.power + 1E-30)  # -300 dB floor

    @cached_property
    def snr_db(self):
        """Measured signal-to-noise ratio of data in dB."""
        return 10 * np.log10(self.snr + 1E-30)  # -300 dB floor

    @cached_property
    def noise_fd(self):
        """Measured noise per sample in time-gated frequency-domain data."""
        tg_length = round(self.t_to_n(np.sum(self.time_gate)) / 2) * 2
        # Separate processing of up- and downchirps
        noise = []
        for sl in [self.sl_upchirps, self.sl_downchirps]:
            if sl is not None:
                fd = np.abs(self.gated_fd_data[sl])
                fd = (fd[0:-1:] - fd[1::])  # Processing of adjacent radar echoes
                noise += [np.mean(fd**2, axis=0)]
        noise = np.mean(noise, axis=0)  # Mean value of both chirp directions
        # Account for processing gains and losses
        noise = noise * self.td_length / tg_length / 2
        return noise

    @cached_property
    def power_fd(self):
        """Measured power per sample in time-gated frequency-domain data."""
        power = np.mean(np.abs(self.gated_fd_data)**2, axis=0)
        power = power - self.noise_fd
        power[power < 0] = 0  # Returns 0 if two consecutive measurements are incoherent (different to power())
        return power

    @cached_property
    def snr_fd(self):
        """Measured signal-to-noise ratio per sample in time-gated frequency-domain data."""
        snr = self.power_fd / self.noise_fd
        return snr

    @cached_property
    def noise_fd_db(self):
        """Measured noise per sample in time-gated frequency-domain data in dB."""
        return 10 * np.log10(self.noise_fd + 1E-30)  # -300 dB floor

    @cached_property
    def power_fd_db(self):
        """Measured power per sample in time-gated frequency-domain data in dB."""
        return 10 * np.log10(self.power_fd + 1E-30)  # -300 dB floor

    @cached_property
    def snr_fd_db(self):
        """Measured signal-to-noise ratio per sample in time-gated frequency domain data in dB."""
        return 10 * np.log10(self.snr_fd + 1E-30)  # -300 dB floor

    @cached_property
    def tof(self):
        """Measured time-of-flight in seconds."""
        # Estimate time-domain parameter
        roi = np.asarray(np.rint(self.t_to_n(np.asarray(self.roi))), int) if self.roi is not None else None
        n, phi = {
            'basic': lambda: basic_estimator(self.td_data_abs, self.td_data_rad, roi),
            'QI': lambda: qi_estimator(self.td_data_abs, self.td_data_rad, roi),
            'QIPS': lambda: qips_estimator(self.window, self.td_data_abs, self.td_data_rad, roi),
            'zoom': lambda: zoom_estimator(self.fd_data,
                                           roi,
                                           self.estimator.get('n_rec', 1) if type(self.estimator) is dict else 1),
        }[self.estimator.get('mode') if type(self.estimator) is dict else self.estimator]()
        tau = self.n_to_t(n)

        # Apply near-field correction
        if self.nearfield_correction is not None:
            if 'rtt_offset' in self.nearfield_correction:
                rtt_uncompensated = tau - self.nearfield_correction['rtt_offset']
            else:
                rtt_uncompensated = tau
            rtt_distance_uncompensated = rtt_uncompensated / 2 * C0 / self.refractive_index
            tau = tau - self.get_pulse_position_variation(rtt_distance_uncompensated)
            phi = phi - self.get_pulse_phase_variation(rtt_distance_uncompensated)

        # Unwrap phase in case of a compensated IF path phase response
        if self.if_path_filter in ['amp+phase', 'phase']:
            phi_tau = -tau * 2 * np.pi * self.center_freq
            phi = phi + np.round((phi_tau - phi) / (2 * np.pi)) * 2 * np.pi  # Unwrap on 2pi interval
            self.delta_phi = phi - phi_tau  # Deviation of phase between pulse phase and pulse position measurement

        # Combine up- and downchirps
        if self.use_triangular_modulation:
            if self.use_sweep_interleaving:
                # Insert latest estimates from previous computation
                tau = np.insert(tau, 0, self.tau_mem) if self.tau_mem is not None else tau
                phi = np.insert(phi, 0, self.phi_mem) if self.phi_mem is not None else phi
                # Store latest estimates
                self.tau_mem = tau[-1]
                self.phi_mem = phi[-1]
                # Interleaved computation of tau and phi
                tau = np.convolve(tau, [0.5, 0.5], mode='valid')
                phi = np.convolve(phi, [0.5, 0.5], mode='valid')
            else:
                # Computation of tau and phi without interleaving of every adjacent pair of estimates
                tau = (tau[self.sl_upchirps] + tau[self.sl_downchirps]) / 2
                phi = (phi[self.sl_upchirps] + phi[self.sl_downchirps]) / 2

        # Unwrap phase in case of an uncompensated IF path phase response
        if self.if_path_filter not in ['amp+phase', 'phase']:
            phi_tau = -tau * 2 * np.pi * self.center_freq
            phi = phi + np.round((phi_tau - phi) / np.pi) * np.pi  # Unwrap on 1pi interval
            self.delta_phi = phi - phi_tau  # Deviation of phase between pulse phase and pulse position measurement

        # Use pulse phase instead of pulse position
        if self.use_phase:
            tau = -phi / (2 * np.pi * self.center_freq)

        return tau

    @cached_property
    def distance(self):
        """Measured distance in meter."""
        rtt_distance = self.tof / 2 * C0 / self.refractive_index
        return self.sign * (rtt_distance + self.origin) if self.origin is not None else self.sign * rtt_distance

    @cached_property
    def refractive_index(self):
        """Measured refractive index of atmosphere."""
        refractivity = {
            None: lambda: 0,
            'MPM93': lambda: np.real(mpm93(self.center_freq, self.temp_data, self.press_data, self.hum_data)),
            'S&W53': lambda: smith_weintraub1953(self.temp_data, self.press_data, self.hum_data),
            'EQ1': lambda: eq1(self.center_freq, self.temp_data, self.press_data, self.hum_data, self.co2_conc_data),
            'EQ2': lambda: eq2(self.center_freq, self.temp_data, self.press_data, self.hum_data, self.co2_conc_data),
            'EQ3': lambda: eq3(self.center_freq, self.temp_data, self.press_data, self.hum_data, self.co2_conc_data),
            'dband': lambda: eq2(self.center_freq, self.temp_data, self.press_data, self.hum_data, self.co2_conc_data),
        }[self.refractive_index_model]()
        return 1 + refractivity * 1E-6

    def get_pulse_position_variation(self, rtt_distance):
        """Return of pulse position variation due to near-field effects of free-space wave propagation."""
        params = self.nearfield_correction
        var = {
            'AM': lambda: approximate_model(rtt_distance, params['d1'], params['d2']) * 2 / C0,
            'AM2': lambda: approximate_model_2_position(
                rtt_distance, params['d1'], params['d2'], 2 * np.pi * self.center_freq / C0),
            'PM': lambda: gen_parametric_model(rtt_distance, params['a_tot'], params['r_off']) * 2 / C0,
            'func': lambda: params['pulse_position_variation_func'](rtt_distance),
        }[params['mode']]()
        return var

    def get_pulse_phase_variation(self, rtt_distance):
        """Return of pulse phase variation due to near-field effects of free-space wave propagation."""
        params = self.nearfield_correction
        var = {
            'AM': lambda: (-approximate_model(rtt_distance, params['d1'], params['d2'])
                           * 4 * np.pi * self.center_freq / C0),
            'AM2': lambda: approximate_model_2_phase(
                rtt_distance, params['d1'], params['d2'], 2 * np.pi * self.center_freq / C0),
            'PM': lambda: (-gen_parametric_model(rtt_distance, params['a_tot'], params['r_off'])
                           * 4 * np.pi * self.center_freq / C0),
            'func': lambda: params['pulse_phase_variation_func'](rtt_distance),
        }[params['mode']]()
        return var

    def do_rf_path_calibration(self, tof0=None):
        """'Calibrate' the RF path on measured data, i.e., initialize the signal filter for the rf path."""
        # Safe old state
        window = self.window
        estimator = self.estimator
        nearfield_correction = self.nearfield_correction
        self.invalidate_cache()
        # Reset parameters
        self.rf_path_response = None  # Reset current rf path response
        self.window = 'boxcar'
        self.estimator = 'basic'
        self.nearfield_correction = None
        # Compute RF-path response
        fd = self.gated_fd_data
        fd = np.mean(fd, axis=0)
        if tof0 is None:
            tof0 = np.mean(self.tof)  # Compute TOF as mean TOF of measurements
        fd = fd * np.exp(1j * 2 * np.pi * self.freq_axis * tof0)
        fd = fd / np.sqrt(np.mean(np.abs(fd)**2))  # Normalize
        self.rf_path_response = fd
        # Restore old state
        self.window = window
        self.estimator = estimator
        self.nearfield_correction = nearfield_correction
        self.invalidate_cache()
        # Clear interleaving memory
        self.tau_mem = None
        self.phi_mem = None

    def load_rf_path_response(self, freq, response):
        """Load the signal filter for the RF path."""
        # Interpolate
        real = np.interp(self.freq_axis, freq, np.real(response))
        imag = np.interp(self.freq_axis, freq, np.imag(response))
        self.rf_path_response = real + 1j * imag

    def load_if_path_response(self, freq, response):
        """Load the signal filter for the IF path."""
        # Interpolate
        real = np.interp(self.if_axis, freq, np.real(response))
        imag = np.interp(self.if_axis, freq, np.imag(response))
        self.if_path_response = real + 1j * imag

    def set_as_origin(self):
        """Set the measured distance as origin."""
        distance = np.mean(self.distance)
        self.origin = -self.sign * distance + self.origin if self.origin is not None else -self.sign * distance
        del self.__dict__['distance']  # Invalidate cache

    def update_if_data(self, if_data, clear_interleaving_mem=False):
        """Update measured if-data.

        Parameters:
        - if_data -- measured real-valued if-data of dimension MxN
        """
        self.if_data = np.asarray(if_data)
        if self.if_data.ndim == 1:
            self.if_data = np.asarray([if_data])
        if clear_interleaving_mem:
            self.tau_mem = None
            self.phi_mem = None

        self.invalidate_cache()

    def update_atmospheric_data(self, temp_data=None, press_data=None, hum_data=None, co2_conc_data=None):
        """Update measured atmospheric data.

        Parameters:
        - temp_data: air temperture (K)
        - press_data: air pressure (Pa)
        - hum_data: humidity (%)
        - co2_conc_data: carbon dioxide concentration (0...1)"""
        # Mean values if input data are arrays
        if temp_data is not None:
            self.temp_data = np.mean(temp_data)
        if press_data is not None:
            self.press_data = np.mean(press_data)
        if hum_data is not None:
            self.hum_data = np.mean(hum_data)
        if co2_conc_data is not None:
            self.co2_conc_data = np.mean(co2_conc_data)

        self.invalidate_cache()

    def invalidate_cache(self):
        """Invalidate cache."""
        cached_properties = [
            'td_data',
            'td_data_abs',
            'td_data_db',
            'td_data_rad',
            'gated_td_data',
            'gated_td_data_abs',
            'gated_td_data_db',
            'gated_td_data_rad',
            'fd_data',
            'gated_fd_data',
            'noise',
            'power',
            'snr',
            'noise_db',
            'power_db',
            'snr_db',
            'noise_fd',
            'power_fd',
            'snr_fd',
            'noise_fd_db',
            'power_fd_db',
            'snr_fd_db',
            'tof',
            'distance',
            'refractive_index',
        ]
        for cached_property_ in cached_properties:
            with suppress(KeyError):
                del self.__dict__[cached_property_]

    @property
    def sl_upchirps(self):
        """Slice of upchirp if-data."""
        return {'uu': np.s_[:], 'dd': None, 'ud': np.s_[0:-1:2], 'du': np.s_[1::2]}[self.if_data_order]

    @property
    def sl_downchirps(self):
        """Slice of downchirp if-data."""
        return {'uu': None, 'dd': np.s_[:], 'ud': np.s_[1::2], 'du': np.s_[0:-1:2]}[self.if_data_order]

    @property
    def sign(self):
        """Sign of measured distance (1: positive, -1: negative)."""
        return {'positive': 1, 'negative': -1}[self.direction]
