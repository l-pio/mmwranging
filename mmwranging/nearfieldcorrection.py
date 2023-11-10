import numpy as np
from numpy.linalg import norm


C0 = 299792458
Z0 = 376.730313667


def approximate_model(distance, d1, d2):
    """Calculate the distance variation using the approximate model."""
    return (d1**2 + d2**2) / (16 * distance)


def gen_parametric_model(distance, a_tot, r_off):
    """Calculate the distance variation using the generalized parametric model."""
    return a_tot / (4 * np.pi * (distance + r_off))


def gen_parametric_model_fit(r, data, max_deviation):
    """Fit parameters k1, and k2 of the generalized parametric model to measured or simulated data."""
    # Sort r / data with increasing r
    idx_array = np.argsort(r)
    _r = r[idx_array]
    _data = data[idx_array]
    # Iterate through r from high to low until termination
    # Idea: curve-fit in the linear region of the PPV's inverse
    k1 = k2 = r_min = None
    for idx in range(2, len(r)):
        # Linear fit of inverted data
        x = _r[-idx:]
        y = 1 / _data[-idx:]
        [a, b] = np.polyfit(x, y, 1)
        # Terminate
        deviation = max(np.abs(1 / (a * x + b) - _data[-idx:]))
        if deviation > max_deviation:
            break
        # Compute return parameter
        k1 = 2 * np.pi / a
        k2 = k1 * b / (2 * np.pi)
        r_min = _r[-idx]
    return k1, k2, r_min


class PPVSimulator:
    """
    Simulate pulse position variation, and pulse phase variation by means of physical optics.

    Notes:
    - The reference point of the antenna (P1) is always (0,0,0) and the antenna aperture is in the xy-plane.
    - The target meshgrid is referenced to the reference point of the target (P2)
    """
    def __init__(self, **kwargs):
        """Initialize a new simulation instance."""
        # Antenna & target
        self.antenna_gain = kwargs.get('antenna_gain', None)
        self.target_rcs = kwargs.get('target_rcs', None)
        # Aperture illumination function of the transmitting antenna & Meshgrids
        self.freqs = None
        self.aif = None
        self.meshgrid_spacing = 0.5  # In wavelengths
        self.meshgrid_ant = None
        self.meshgrid_tar = None
        self.meshgrid_ant_density = None  # Area per meshcells
        self.meshgrid_tar_density = None  # Area permeshcells
        self.ant_z_offset = None  # Offset from origin to antenna plane
        self.r0 = None
        self.n1 = None
        self.n2 = None
        # Simulation results
        self.reflection_coefficient = None
        self.phase_variation = None
        self.pulse_phase_variation = None
        self.pulse_position_variation = None
        self.amplitude_gain = None
        self.pulse_amplitude_gain = None

    @staticmethod
    def generate_meshgrid(shape, dimension, spacing):
        """Generate a 3D meshgrid in the xy-plane at z=0."""
        # Generate generic rectangular meshgrid
        n = int(np.ceil((np.max(dimension) / spacing + 1) / 2) * 2)
        xy = np.linspace(-np.max(dimension) / 2, np.max(dimension) / 2, n)
        z = 0
        meshgrid = np.array(np.meshgrid(xy, xy, z, indexing='ij')).reshape([3, -1]).T

        # Adjust shape of the meshgrid
        meshgrid = {
            'circular': lambda: meshgrid[norm(meshgrid, axis=-1) <= dimension / 2],
            'square': lambda: meshgrid[(np.abs(meshgrid[:, 0]) <= dimension / 2) and
                                       (np.abs(meshgrid[:, 1]) <= dimension / 2)],
            'rectangular': lambda: meshgrid[(np.abs(meshgrid[:, 0]) <= dimension[0] / 2) and
                                            (np.abs(meshgrid[:, 1]) <= dimension[1] / 2)]
        }[shape]()

        # Compute density of meshgrid (meshcells per area)
        meshcells = meshgrid.shape[0]
        density = meshcells / {
            'circular': lambda: np.pi * dimension**2 / 4,
            'square': lambda: dimension**2,
            'rectangular': lambda: dimension[0] * dimension[1]
        }[shape]()

        return meshgrid, density

    def load_aif_from_cst_export(self, freqs, filenames, shape, dimension, translation=None, length_unit='mm'):
        """Load transmitted E-field on antenna aperture plane (xy-plane) from CST export and init meshgrid."""
        # Load data from files
        coordinates = []
        efields = []
        for freq, filename in zip(freqs, filenames):
            data = np.genfromtxt(filename, skip_header=2)
            coordinates += [
                np.array([data[:, 0], data[:, 1], data[:, 2]]).T / {'m': 1, 'cm': 1E2, 'mm': 1E3}[length_unit]
            ]
            efields += [
                np.array([data[:, 3] + 1j * data[:, 4], data[:, 5] + 1j * data[:, 6], data[:, 7] + 1j * data[:, 8]]).T
            ]

        # Check whether coordinates of different AIF simulations are identical
        for idx in range(1, len(coordinates)):
            if not np.allclose(coordinates[0], coordinates[idx]):
                raise ValueError('Coordinates of different AIF simulations are not identical!')
        coordinates = np.asarray(coordinates[0])
        efields = np.asarray(efields)

        # Crop to certain shape
        crop_mask = {
            'circular': lambda: np.sqrt(coordinates[:, 0]**2 + coordinates[:, 1]**2) <= dimension / 2,
            'square': lambda: ((np.abs(coordinates[:, 0]) <= dimension / 2) and
                               (np.abs(coordinates[:, 1]) <= dimension / 2)),
            'rectangular': lambda: ((np.abs(coordinates[:, 0]) <= dimension[0] / 2) and
                                    (np.abs(coordinates[:, 1]) <= dimension[1] / 2))
        }[shape]()
        coordinates = coordinates[crop_mask]
        efields = np.array([efield[crop_mask] for efield in efields])

        # Area
        area = {
            'circular': lambda: np.pi * dimension**2 / 4,
            'square': lambda: dimension**2,
            'rectangular': lambda: dimension[0] * dimension[1]
        }[shape]()

        # Translate coordinates
        if translation is not None:
            coordinates = coordinates + np.asarray(translation)

        # init AIF
        self.init_aif(freqs, coordinates, efields, area)

    def init_aif(self, freqs, coordinates, efields, area):
        """Initialize transmitted E-fields on antenna aperture plane (xy-plane) and initialize meshgrid."""
        # Check whether E-fields are in the xy-plane
        for idx in range(1, len(coordinates)):
            if not np.allclose(coordinates[0][2], coordinates[idx][2]):
                raise ValueError('Simulated E-fields are not in the xy-plane!')

        # If the antenna-aperture plane does not pass through the origin: save z-offset
        self.ant_z_offset = coordinates[0][2]

        # Set data
        self.freqs = np.asarray(freqs)
        self.meshgrid_ant = np.asarray(coordinates)  # Set meshgrid on AIF coordinates
        self.aif = np.asarray(efields)  # Set respective E-fields as AIFs
        self.n1 = np.array([0, 0, 1])  # Set antenna in xy-plane, radiating into positive z direction
        meshcells = self.meshgrid_ant.shape[0]
        self.meshgrid_ant_density = meshcells / area

    def init_homogeneous_aif(self, freqs, shape, dimension, polarization='x', focal_length=np.inf):
        """Initialize a homogeneously illuminated AIF (xy-plane) and initialize meshgrid."""
        # Compute meshgrid
        wavelength_min = min(C0 / freqs)
        meshgrid, meshgrid_density = self.generate_meshgrid(shape, dimension, self.meshgrid_spacing * wavelength_min)
        area = meshgrid.shape[0] / meshgrid_density

        # Set homogeneous amplitude distribution with respect to the polarization
        aif = np.ones([len(freqs), len(meshgrid), 3], dtype=complex) * {'x': [1, 0, 0], 'y': [0, 1, 0]}[polarization]

        # Compute phase distribution for the given focal length
        # focal_length=np.inf: hom. phase, focal_length > 0: focused aperture, focal_length < 0: diverged aperture
        if focal_length is not np.inf:
            path_difference = norm(meshgrid - np.asarray([0, 0, focal_length]), axis=-1) - abs(focal_length)
            sign = 1 if focal_length >= 0 else -1
            for idx, k in enumerate(2 * np.pi * freqs / C0):
                aif[idx] = (aif[idx].T * np.exp(1j * sign * k * path_difference)).T

        # Initialize AIF
        self.init_aif(freqs, meshgrid, aif, area)

    def init_target_meshgrid(self, shape, dimension, position=None):
        """Initialize the target meshgrid in the xy-plane at a given position."""
        # Compute meshgrid
        wavelength_min = min(C0 / self.freqs)
        self.meshgrid_tar, self.meshgrid_tar_density = self.generate_meshgrid(
            shape, dimension, self.meshgrid_spacing * wavelength_min
        )

        # Set position and orientation
        if position is not None:
            self.r0 = np.asarray(position)
        self.n2 = np.array([0, 0, -1])  # xy-plane, scattering into negative z direction

    def set_target_position(self, position):
        """Set the absolute position of the target (P1)."""
        self.r0 = np.asarray(position)

    def rotate_target_meshgrid(self, alpha=0, beta=0, gamma=0):
        """Rotate the target meshgrid by euler angles."""
        # Rotation matrix
        ca = np.cos(alpha)
        cb = np.cos(beta)
        cg = np.cos(gamma)
        sa = np.sin(alpha)
        sb = np.sin(beta)
        sg = np.sin(gamma)
        rot = np.array([
            [ca * cg - sa * cb * sg, -ca * sg - sa * cb * cg, sa * sb],
            [sa * cg + ca * cb * sg, -sa * sg + ca * cb * cg, -ca * sb],
            [sb * sg, sb * cg, cb]
        ])
        # Rotate meshgrid and normal vector
        self.meshgrid_tar = np.array([np.matmul(rot, element) for element in self.meshgrid_tar])
        self.n2 = np.matmul(rot, self.n2)

    def translate_target_meshgrid(self, x=0, y=0, z=0):
        """Translate the target meshgrid."""
        trans = np.array([x, y, z])
        self.meshgrid_tar = np.array([element + trans for element in self.meshgrid_tar])

    def start_simulation(self):
        """Start the simulation, where the simulation results are saved as class attributes."""
        # Position vectors with dimension [meshgrid_ant.shape[0], meshgrid_tar.shape[0], 3]
        r1 = np.empty([self.meshgrid_ant.shape[0], self.meshgrid_tar.shape[0], 3])
        r2 = np.empty([self.meshgrid_ant.shape[0], self.meshgrid_tar.shape[0], 3])
        for idx in range(3):
            r1[..., idx], r2[..., idx] = np.meshgrid(
                self.meshgrid_ant[:, idx], self.meshgrid_tar[:, idx], indexing='ij'
            )
        r = -r1 + self.r0 + r2
        er = (r.T / norm(r, axis=-1).T).T

        # Loop through frequencies: PO simulation
        self.phase_variation = np.empty(self.freqs.shape)
        self.amplitude_gain = np.empty(self.freqs.shape)
        for idx, (e_tr, freq) in enumerate(zip(self.aif, self.freqs)):
            # Equivalent surface current density on the transmitting antenna aperture
            m_ant = 2 * np.cross(e_tr, self.n1)
            m_ant_ = np.repeat(m_ant[:, np.newaxis, :], self.meshgrid_tar.shape[0], axis=1)

            # Wavenumber and greens function
            k = 2 * np.pi * freq / C0
            g = np.exp(-1j * k * norm(r, axis=-1)) / (4 * np.pi * norm(r, axis=-1))
            g_ = np.repeat(g[:, :, np.newaxis], 3, axis=2)

            # Equivalent electric surface current density on the target with dimension meshgrid_tar.shape
            integrand = np.cross(er, np.cross(er, m_ant_)) * g_
            integral = np.sum(integrand, axis=0) / self.meshgrid_tar_density
            j_tar = np.cross(2 * 1j * k * self.n2 / Z0, integral)
            j_tar_ = np.repeat(j_tar[np.newaxis, :, :], self.meshgrid_ant.shape[0], axis=0)

            # Receiving E-field on the antenna with dimension meshgrid_ant.shape
            integrand = np.cross(er, np.cross(er, j_tar_)) * g_
            integral = np.sum(integrand, axis=1) / self.meshgrid_ant_density
            e_re = 1j * k * Z0 * integral

            # Reflection coefficient
            integrand_num = np.array([np.vdot(a, b) for a, b in zip(e_tr, e_re)])
            integral_num = np.sum(integrand_num) / self.meshgrid_ant_density
            integrand_den = np.array([np.vdot(b, b) for b in e_tr])
            integral_den = np.sum(integrand_den) / self.meshgrid_ant_density
            self.reflection_coefficient = -integral_num / integral_den

            # Reflection coefficient of reference
            distance = norm(self.r0 - [0, 0, self.ant_z_offset])
            phase_term = -np.exp(-1j * k * 2 * distance)
            if self.antenna_gain is not None and self.target_rcs is not None:
                # Taking the amplitude term into account
                antenna_gain_lin = 10**(self.antenna_gain / 10)  # dBi to linear units
                target_rcs_lin = 10**(self.target_rcs / 10)  # dBsm to linear units
                wavelength = C0 / freq
                amplitude_term = antenna_gain_lin * wavelength / distance**2 * np.sqrt(target_rcs_lin / (4 * np.pi)**3)
            else:
                # Taking the amplitude term not into account
                amplitude_term = 1
            reflection_coefficient_ref = amplitude_term * phase_term

            # Phase and amplitude variation
            quotient = self.reflection_coefficient / reflection_coefficient_ref
            self.phase_variation[idx] = np.angle(quotient)
            self.amplitude_gain[idx] = 20 * np.log10(np.abs(quotient))  # In dB units

        # Numerical computation of pulse properties
        center_freq = np.mean(self.freqs)
        y = np.array([self.phase_variation]).T
        a = np.array([np.ones(self.freqs.size), 2 * np.pi * (center_freq - self.freqs)]).T
        res = np.matmul(np.matmul(np.linalg.inv(np.matmul(a.T, a)), a.T), y)

        self.pulse_phase_variation = res[0, 0]
        self.pulse_position_variation = res[1, 0]
        self.pulse_amplitude_gain = 20 * np.log10(np.mean(10**(self.amplitude_gain / 20)))  # In dB units
