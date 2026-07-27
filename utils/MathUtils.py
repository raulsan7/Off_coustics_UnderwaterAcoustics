"""
Module: MathUtils.py
Description: Mathematical utility functions for signal processing, spectral analysis,
             and geometric operations in underwater vibroacoustic modeling.

             ┌────────────────────────────────────────────────────────┐
             │                     MathUtils.py                       │
             ├────────────────────────────────────────────────────────┤
             │  • remove_duplicate_nodes  ──> Geometry cleaning        │
             │  • compute_rfft            ──> RFFT processing          │
             │  • alpha_hankel            ──> Hankel wave correction   │
             │  • divide_span             ──> Discretization sizing    │
             │  • generate_timeseries...  ──> Signal reconstruction    │
             │  • filter_non_usefull_freqs──> Spectral mask building   │
             └────────────────────────────────────────────────────────┘

Author: Raul Sanz Ramirez (raul.sanz.ramirez@upm.es / raul.sanz.ramirez@gmail.com)
Institution: Universidad Politecnica de Madrid - ETSIAE
Date: 07/2026 
"""

import numpy as np
from scipy.special import hankel2


def remove_duplicate_nodes(pos_0      : np.ndarray = None,      # [m] Nodes coordinates shape(Nnodes, 3)
                           acc        : np.ndarray = None,      # [m/s^2] Acceleration array shape(nfreqs, Nnodes, 3)
                           tol        : float      = 1e-4,      # [-] Distance Tolerance
                           delete_last: bool       = False):    # [-] Flag to delete last node of each members except last
    """
    Cleans structural joints by removing duplicate adjacent nodes between consecutive members.

    Geometrical Joint Concept:
    ┌───────────┐               ┌───────────┐
    │ Member i  │ ──●     ●──   │ Member i+1│  ===> Coincident joint node detected
    └───────────┘   last  first └───────────┘       (removed via mask)

    Parameters:
    - pos_0       [ndarray] (Nmembers, Nnodes, 3) : Spatial node positions.
    - acc         [ndarray] (nt, Nmembers*Nnodes, 3) : Accelerations (optional).
    - tol         [float]                         : Joint proximity tolerance (default: 1e-4 m).
    - delete_last [bool]                          : Force removal of the final structural node.

    Returns:
    - mask [ndarray] (Nmembers * Nnodes,) : Boolean mask (True = Keep, False = Duplicate).
    """

    Nmembers, Nnodes, _ = pos_0.shape
    nt = acc.shape[0] if acc is not None else None
    # Build mask: start with all True
    mask = np.ones(Nmembers * Nnodes, dtype=bool)

    for i in range(Nmembers - 1):
        last_node_i     = pos_0[i,  -1, :]   # last node of member i
        first_node_next = pos_0[i+1, 0, :]   # first node of member i+1

        if np.all(np.abs(last_node_i - first_node_next) < tol):
            flat_idx = i * Nnodes + (Nnodes - 1)  # index in flattened array
            mask[flat_idx] = False

    if delete_last:
        mask[-1] = False

    n_removed = np.sum(~mask)
    # print(f"Removed {n_removed} duplicate node(s) at flat indices: {np.where(~mask)[0].tolist()}")

    # # Apply mask to positions
    # pos_0_flat  = pos_0.reshape(Nmembers * Nnodes, 3)
    # pos_0_clean = pos_0_flat[mask]

    # # Apply mask to accelerations if provided
    # acc = acc.reshape((nt, Nmembers*Nnodes,3)) if acc is not None else None
    # acc_clean = acc[:, mask, :] if acc is not None else None

    return mask

def compute_rfft(array      : np.ndarray = None,    # [-] Array to convert into frequency domain
                 nt         : int        = None,    # [-] Number of timesteps
                 dt         : float      = None,    # [s] Timestep size
                 skipf      : int        = 1,       # [-] Skips frequency data e.g. Freqs[::skipf]
                 remove_zero: bool       = True):   # [-] Wheter to remove mean value
    """
    Computes the Real Fast Fourier Transform (RFFT) of a 3D time-series array.

    Processing Pipeline:
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │ Subtract     │ ──> │ Apply        │ ──> │ Compute      │ ──> │ Decimate /   │
    │ Mean         │     │ Hanning Win. │     │ Scaling & FFT│     │ Filter f=0   │
    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘

    Parameters:
    • array       [ndarray] (nt, Nnodes, 3) : Time-domain physical inputs.
    • nt          [int]                     : Number of time steps.
    • dt          [float] (s)               : Time step size.
    • skipf       [int]                     : Frequency decimation factor (keep every N-th).
    • remove_zero [bool]                    : Drop the DC frequency component ($f = 0$).

    Returns:
    • f     [ndarray] (nf, Nnodes, 3) : Frequency-domain complex spectrum.
    • freqs [ndarray] (nf,)           : Evaluated frequency bins [Hz].
    """

    window = np.hanning(nt)
    win_norm = np.mean(window)
    f = array.copy()
    f -= np.mean(f, axis=0, keepdims=True)        # Substract mean not to get the f=0 peak
    f *= window[:,None,None]
    f = 2*np.fft.rfft(f, axis=0)/(nt*win_norm)
    freqs = np.fft.rfftfreq(nt, d=dt)

    if skipf > 1:
        print(f'Number of frequencies before skipf: {len(freqs)}')
        f = f[::skipf,:,:]
        freqs = freqs[::skipf]

    if remove_zero:
        mask = freqs > 0.0
        freqs = freqs[mask]
        f = f[mask,:,:]

    return f, freqs

def alpha_hankel(k: np.ndarray = None,      # [1/m] Wavenumber array shape(nfreqs)
                 D: float      = None):     # [m] Diameter
    """
    Analytical scattering correction coefficient $\alpha$ based on Hankel functions.

    This implements the analytical shielding factor for a cylindrical structure:
    $$\alpha = -\frac{3i}{\pi (ka)^2 H_1^{(2)\prime}(ka)}$$
    where the derivative of the second-kind Hankel function of order 1 is calculated as:
    $$H_1^{(2)\prime}(x) = \frac{1}{2} \left[ H_0^{(2)}(x) - H_2^{(2)}(x) \right]$$

    Parameters:
    - k [ndarray] (1/m) : Acoustic wavenumber $k = \omega / c$.
    - D [float] (m)     : Cylindrical monopile outer diameter.

    Returns:
    - alpha [ndarray] (nf,) : Complex frequency-dependent correction factors.
    """

    ka   = k * D / 2.0
    H0   = hankel2(0, ka)
    H2   = hankel2(2, ka)
    H1p  = 0.5 * (H0 - H2)
    return -3j / (np.pi * ka**2 * H1p)   # (nf,)

def divide_span(x: np.ndarray = None):      # [-] Array to divide
    """
    Computes representative span intervals ($dx$) for an ordered spatial vector.

    Calculation Scheme for node i:
    ├─────────●─────────┼─────────●─────────┤
            x[i-1]    inf[i]     x[i]     sup[i]
    │<─── dx[i-1] ─────>│<────── dx[i] ────>│

    Parameters:
    • x [ndarray] : 1D spatial coordinates (e.g., node depths).

    Returns:
    • dx [ndarray] : Segment lengths associated with each grid node.

    Raises:
    • ValueError: If spacing direction is inconsistent (non-monotonic grid).
    """

    flipped = False
    if x[0] > x[-1]:
        x = x[::-1]
        flipped = True

    n = len(x)
    inf = np.zeros(n)
    sup = np.zeros(n)
    dx = np.zeros(n)

    for i in range(n):
        if i == 0:
            inf[i] = x[i]
        else:
            inf[i] = (x[i] + x[i-1])/2

        if i == n-1:
            sup[i] = x[i]
        else:
            sup[i] = (x[i] + x[i+1])/2

        dx[i] = sup[i] - inf[i]

    if flipped:
        dx = dx[::-1]

    if not(all(dx>0) or all(dx<0)):
        raise ValueError("Error on divide_span: check input")

    return dx

def generate_timeseries_banded_sines(peaks     : np.ndarray = None,     # [-] Frequencies where peaks are located
                                     keys      : list       = None,     # [-] List with labels
                                     t         : np.ndarray = None,     # [s] Time array shape(nt, )
                                     zeta      : float      = 0.02,     # [-] Structural damping
                                     nfreq     : int        = 50,       # [-] Number of frequencies in gaussian clustering
                                     seed      : int        = 42,       # [-] Seed to randomize phases
                                     used_freqs: bool       = False,    # [Hz] Used frequencies
                                     fcut      : float      = 10.0):    # [Hz] Where to start clustering
    """
    Reconstructs a time-series signal from spectral peak distributions.

    Signal Types:
    - Pure Tones (Shaft harmonics):
      $$a(t) = \sqrt{2} A_{\text{rms}} \sin(2\pi f_0 t + \phi)$$
    - Banded Noise (Gear meshes / high-frequency):
      Approximated via $N$ Gaussian-distributed discrete sine lines centered around $f_0$:
      $$\sigma = \zeta f_0$$

    Parameters:
    - peaks      [ndarray] (N, 2) : Col 0: frequency [Hz], Col 1: RMS amplitude.
    - keys       [list of str]    : Component labels (e.g., 'mesh' vs 'shaft').
    - t          [ndarray] (s)    : Target time vector.
    - zeta       [float]          : Modal damping ratio governing bandwidth.
    - nfreq      [int]            : Spectral lines per Gaussian band.
    - seed       [int]            : Random seed for phase generation.
    - used_freqs [bool]           : If True, also outputs the synthesized frequencies.
    - fcut       [float] (Hz)     : Cutoff frequency to force banded reconstruction.

    Returns:
    - a     [ndarray]     : Synthesized physical time-series.
    - freqs [ndarray] (M) : Sorted unique active frequencies (only if used_freqs is True).
    """
    
    rng = np.random.default_rng(seed)
    t   = np.asarray(t)
    a   = np.zeros_like(t)

    freq_list = []

    for (f0, Arms), key in zip(peaks, keys):

        # -----------------------------
        # GEAR MESH → banded Gaussian
        # -----------------------------
        if "mesh" in key or f0 > fcut:

            sigma = zeta * f0

            if sigma <= 0:
                continue

            fk = np.linspace(f0 - 4*sigma, f0 + 4*sigma, nfreq)

            g = np.exp(-0.5 * ((fk - f0) / sigma)**2)
            g /= np.sqrt(np.sum(g**2))

            Arms_k = Arms * g
            phases = rng.uniform(0, 2*np.pi, nfreq)

            a += np.sum(
                Arms_k[:, None] *
                np.sin(2*np.pi*fk[:, None]*t[None, :] + phases[:, None]),
                axis=0
            )

            if used_freqs:
                freq_list.extend(fk)

        # -----------------------------
        # SHAFT ORDERS → pure tone
        # -----------------------------
        else:
            phase = rng.uniform(0, 2*np.pi)
            a += np.sqrt(2) * Arms * np.sin(2*np.pi*f0*t + phase)

            # 👉 guardar frecuencia
            if used_freqs:
                freq_list.append(f0)

    if used_freqs:
        # opcional: eliminar duplicados y ordenar
        freqs = np.unique(np.array(freq_list))
        return a, freqs

    return a

def filter_non_usefull_freqs(freqs        : np.ndarray = None,  # [Hz] Frequency array shape(nfreqs,)
                             mantain_freqs: np.ndarray = None,  # [Hz] Frequency list to maintain
                             freqs_over   : float      = None,  # [Hz] Lower bound
                             freqs_under  : float      = None): # [Hz] Upper bound
    """
    Generates a frequency-mask to filter out non-essential spectral bins.

    Retains frequency bins within one resolution step ($\pm df$) of targeted peaks,
    optionally bounding the mask calculation to specific limits.

    Visual Filtering Zone:
                     df     df
                  |<───> f0 <───>|
    ──────────────■──────■──────■─────────────── (freqs)
                 [Keep] [Keep] [Keep]

    Parameters:
    • freqs         [ndarray]    : Uniformly-spaced frequency grid.
    • mantain_freqs [ndarray]    : Targeted physical excitation frequencies to preserve.
    • freqs_over    [float] (Hz) : Lower limit of the filtering window (optional).
    • freqs_under   [float] (Hz) : Upper limit of the filtering window (optional).

    Returns:
    • mask [ndarray] (bool) : Boolean index mask matching the size of freqs.
    """
    
    freqs = np.asarray(freqs)
    mantain_freqs = np.asarray(mantain_freqs)
    
    # compute frequency resolution (bin size)
    df = freqs[1] - freqs[0]

    #  check if freqs is equally spaced
    if not np.allclose(np.diff(freqs), df):
        raise ValueError("Input freqs must be equally spaced.")
    
    # Conserve all by default
    mask = np.ones_like(freqs, dtype=bool)
    
    # Where is filter applied
    filter_zone = np.ones_like(freqs, dtype=bool)
    if freqs_over is not None:
        filter_zone &= (freqs >= freqs_over)
    if freqs_under is not None:
        filter_zone &= (freqs <= freqs_under)

    mask[filter_zone] = False  # Start with all False in the filter zone

    for f in mantain_freqs:
        near = (np.abs(freqs-f) <= df) & filter_zone
        mask |= near
    
    return mask

def integral_elipse_z(Dy: float = 10.0,     # [m] Diameter in y-direction
                      Dz: float = 12.5,     # [m] Diameter in z-direction
                      k : float = 0.0):     # [m] z(y) = k line
    """
    Area under z = k inside ellipse:
    (2y/Dy)^2 + (2z/Dz)^2 <= 1
    """

    a = Dy / 2.0
    b = Dz / 2.0

    elipse_area = np.pi * a * b

    k = np.clip(k, -b, b)

    u = k / b

    return 0.5 * elipse_area + a * b * (np.arcsin(u) + u * np.sqrt(1.0 - u*u))


