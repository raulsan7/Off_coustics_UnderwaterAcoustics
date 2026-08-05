import numpy as np
import matplotlib.pyplot as plt
import plotting.plot_utils as pu
import utils.AcousticUtils as au
import utils.EnvironmentalData as ed
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# ---------- Structural Plots ---------- #
def plot_structure(case   : dict  = None,       # [-] Dictionary containing simulation data
                   mode   : str   = '3D',       # [-] Plot mode for the structure representation
                   WindDir: bool  = True,       # [-] Wheter to plot or not wind direction
                   figsize: tuple = (8,6),      # [-] Figure size
                   ax_in  : plt.Axes = None):   # [-] Optional external axes for subplots
    """
    Plots the structure nodes in shape (Nmembers, Nnodes, 3)

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The generated figure and axes objects for further editing.
    """

    case["has_nodes"] = "Structure_nodes" in case
    if not case["has_nodes"]:
        print("plot_structure(): no structure data, skipping")
        return None, None
    
    if mode not in ('3D', 'xy', 'xz'):
        raise ValueError("mode must be '3D', 'xy', or 'xz'")
    
    # Extract nodes and depth
    nodes = case["Structure_nodes"]
    Depth = case["Depth"]
    tag   = case["Case_type"]

    # Find node with maximum Z coordinate to place WindDir arrow
    flat_nodes = nodes.reshape(-1, 3)
    max_z_node = flat_nodes[np.argmax(flat_nodes[:,2])]
    max_x, max_y, max_z = max_z_node

    if mode == '3D':

        if ax_in is None:
            fig = plt.figure(figsize=figsize)
            ax  = fig.add_subplot(111, projection='3d')
        else:
            ax = ax_in
            fig = ax.figure

        # Compute extents for drawing planes
        xs = nodes[..., 0].ravel()
        ys = nodes[..., 1].ravel()
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        xpad = 0.1 * (xmax - xmin) if xmax > xmin else 1.0
        ypad = 0.1 * (ymax - ymin) if ymax > ymin else 1.0

        # Draw seabed and surface planes
        Xg = np.linspace(xmin - xpad, xmax + xpad, 2)
        Yg = np.linspace(ymin - ypad, ymax + ypad, 2)
        XX, YY = np.meshgrid(Xg, Yg)

        ax.plot_surface(XX, YY, np.full_like(XX, -Depth), color = 'gray', alpha = 0.5, shade = True, label = "Seabed")
        ax.plot_surface(XX, YY, np.full_like(XX, 0.0), color='deepskyblue', alpha=0.25, shade=False, label = "Surface")

        # Plot each member as connected nodes
        for member in nodes:
            ax.plot(member[:, 0], member[:, 1], member[:, 2], '-o', color='k', ms=3)

        # Draw Wind Direction Arrow
        if WindDir is not None:
            arrow_length = (xmax - xmin + 2*xpad) * 0.25
            if arrow_length == 0: arrow_length = .75

            pu.draw_direction_arrow(ax, max_x, max_y, case["WindDir"],
                                  length=arrow_length, z=max_z,
                                  color='red',
                                  label=f'WindDir: {case["WindDir"]}°')
        ax.legend()

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.set_title(tag)
            
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.set_zlim(-Depth, max_z + max(10, abs(max_z)*0.1))

    else:
        
        if ax_in is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = ax_in
            fig = ax.figure

        if mode == "xy":
            xs = nodes[..., 0].ravel()
            ys = nodes[..., 1].ravel()
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            xpad = 0.1 * (xmax - xmin) if xmax > xmin else 1.0
            ypad = 0.1 * (ymax - ymin) if ymax > ymin else 1.0

            # Project onto z = 0 plane
            for member in nodes:
                ax.plot(member[:, 0], member[:, 1], '-o', color='k', ms=3)

            if WindDir is not None:
                arrow_length = (xmax - xmin + 2*xpad) * 0.25
                if arrow_length == 0: arrow_length = .75

                pu.draw_direction_arrow(ax, max_x, max_y, case["WindDir"],
                      length=arrow_length, color='red',
                      label=f'WindDir: {case["WindDir"]}°')
                
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_title(f"{tag} - Top View (XY)")
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)
            ax.set_aspect('equal', adjustable='box')
            ax.legend()

        elif mode == "xz":
            xs = nodes[..., 0].ravel()
            zs = nodes[..., 2].ravel()
            xmin, xmax = xs.min(), xs.max()
            zmin, zmax = zs.min(), zs.max()
            xpad = 0.1 * (xmax - xmin) if xmax > xmin else 1.0
            zpad = 0.1 * (zmax - zmin) if zmax > zmin else 1.0

            ax.axhline(0.0, color='deepskyblue', linestyle='--', linewidth=1.0, alpha=0.6, label='Surface')
            ax.axhline(-Depth, color='gray', linestyle='--', linewidth=1.0, alpha=0.6, label="Seabed")

            for member in nodes:
                ax.plot(member[:, 0], member[:, 2], '-o', color='k', ms=3)

            if WindDir is not None:
                arrow_length = (xmax - xmin + 2*xpad) * 0.25
                if arrow_length == 0: arrow_length = .75

                pu.draw_direction_arrow(ax, max_x, max_z, case["WindDir"],
                      length=arrow_length, color='red',
                      label=f'WindDir: {case["WindDir"]}°')

            ax.set_xlabel('x [m]')
            ax.set_ylabel('z [m]')
            ax.set_title(f"{tag} - Side View (XZ)")
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(zmin - zpad, zmax + zpad)
            ax.legend()

    if ax_in is None:
        fig.tight_layout()
        
    return fig, ax


# ---------- Spectrum Plots ---------- #
def plot_spectrum(case         : dict  = None,      # [-] Dictionary containing simulation data
                  mode         : str   = 'SPL',     # [-] Selects Sound Pressure/Power Level. 'SPL' or 'SWL'
                  absorption   : bool  = True,      # [-] Wheter to apply approximate absorption attenuation
                  octave       : bool  = True,      # [-] 1/3 octave or fine resolution
                  filter_under : float = None,      # [Hz] Minimum frequency threshold
                  filter_over  : float = None,      # [Hz] Maximum frequency threshold
                  do_thresholds: bool  = False,     # [-] Whether to plot regulatory/reference thresholds
                  figsize      : tuple = (10,5),    # [-] Figure size
                  ax_in        : plt.Axes = None):  # [-] Optional external axes for subplots
    """
    Plots the acoustic spectrum.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The generated figure and axes objects for further editing.
    """

    if not case["has_spectrums"]:
        print("plot_spectrum(): no spectrum data, skipping")
        return None, None
    
    tag = case["Case_type"]
    color = pu.get_case_color(case)

    # Extract data
    Freqs     = case["Freqs"]
    Observers = case["Obs_spectrums"]
    p         = case["P_spectrums"]
    AxisPos   = case["AxisPos"]
    p_ref     = case["p_ref"]

    # Distance to turbine axis
    distances = np.sqrt((Observers[:,0] - AxisPos[0])**2 + (Observers[:,1] - AxisPos[1])**2)

    if mode == "SPL":
        spl = au.pressure_to_SPL(p, p_ref, absorption, Freqs, distances)
        ylabel = r"SPL [dB re 1 $\mu$ Pa]"
    else:
        raise RuntimeError("plot_spectrum(): mode SWL not implemented yet")
    

    # Filter frequencies
    mask = np.ones_like(Freqs, dtype=bool)
    if filter_under is not None:
        mask &= (Freqs >= filter_under)
    if filter_over is not None:
        mask &= (Freqs <= filter_over)

    Freqs = Freqs[mask]
    spl   = spl[mask]
    oaspl = au.pressure_to_OASPL(p[mask], p_ref, absorption, Freqs, distances)

    if octave:
        Freqs, spl = au.to_third_octave(Freqs, spl, fmin=Freqs.min(), fmax=Freqs.max())


    # Plot 
    if ax_in is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        ax = ax_in
        fig = ax.figure
        
    linestyles   = pu.get_line_styles()
    num_observers = spl.shape[1] if spl.ndim> 1 else 1
    if num_observers > len(linestyles): print("plot_spectrum(): too many observers, linestyles will be repeated")

    for i in range(num_observers):
        ls = linestyles[i % len(linestyles)]
        y_data = spl[:,i] if spl.ndim > 1 else spl
        ax.semilogx(Freqs, y_data, ls, color=color, label=f"r = {distances[i]:.1f} m")

        print(f"\nDistance: {distances[i]:.1f} m" )
        print(f"{tag}: OASPL = {oaspl[i]:.1f}")
    
    # Animal thresholds
    if do_thresholds:
        thr = ed.Animal_threshold(do_plot=False)
        f_hz = thr['f'] * 1000 # kHz --> Hz

        THRESHOLD_STYLES = [
            (f_hz,                       thr['T_Lf'], ':',  [0.4, 0.4, 0.4], 'HT. Low-frequency'),
            (f_hz,                       thr['T_Mf'], '--', [0.7, 0.7, 0.7], 'HT. Mid-frequency'),
            (f_hz,                       thr['T_Hf'], '-.', [0.6, 0.6, 0.6], 'HT. High-frequency'),
            (f_hz,                       thr['T_PW'], '-',  [0.5, 0.5, 0.5], 'HT. Pinnipeds'),
            (thr['atlantic_herring'][:,0], thr['atlantic_herring'][:,1], '-+', [0.6, 0.6, 0.6], 'HT. Atlantic herring'),
            (thr['Atlantic_cod'][:,0],     thr['Atlantic_cod'][:,1],     '-.', [0.8, 0.8, 0.8], 'HT. Atlantic cod'),
        ]

        for (fx, fy, ls, color, label) in THRESHOLD_STYLES:
                ax.plot(fx, fy, ls, color=color, lw=1.2, alpha=0.6, label=label)

        ax.set_xlim(Freqs.min(), max(Freqs.max(), 1000))


    ax.set_ylabel(ylabel)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_title(tag)
    ax.grid(True, which='both', ls='--', alpha=0.4)

    if do_thresholds:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
        if ax_in is None:
            fig.tight_layout(rect=[0, 0.06, 1, 1])
    else:
        ax.legend()
        if ax_in is None:
            fig.tight_layout()

    return fig, ax


# ---------- Polar Plots ---------- #
def plot_polar(case             : dict  = None,        # [-] Dictionary containing simulation data
               mode             : str   = 'OASPL',     # [-] Magnitude to plot: 'OASPL', 'SPL', or 'ABS'
               target_frequency : float = 0.88,        # [Hz] Target frequency for 'SPL' or 'ABS' modes
               absorption       : bool  = True,        # [-] Whether to apply absorption attenuation
               bands            : list  = [10., 100.], # [Hz] Frequency band limits for OASPL
               figsize          : tuple = (7,7),       # [-] Figure size
               ax_in            : plt.Axes = None):    # [-] Optional external axes for subplots
    """
    Plots polar directivity diagram.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The generated figure and axes objects for further editing.
    """

    if not case["has_polar"]:
        print("plot_polar_oaspl(): no polar data, skipping")
        return None, None
    
    tag    = case["Case_type"]
    colors = pu.get_band_colors()

    # Extract data
    theta      = case["Theta_deg_polar"]   
    p          = case["P_polar"]
    Freqs      = case["Freqs"]
    p_ref      = case["p_ref"]
    r          = case["R_polar"]
    Observers  = case["Obs_polar"]
    cx, cy, cz = case["Center_polar"][0], case["Center_polar"][1], case["Z_polar"]

    theta_rad  = np.deg2rad(theta)
    fmin, fmax = Freqs.min(), max(Freqs.max(), 1000)
    N_obs      = len(Observers)
    
    # Filter frequencies within specified bands
    if not bands:
        band_labels = [f"All ({fmin:.2g}-{fmax:.2g} Hz)"]
        band_masks  = [np.ones(len(Freqs), dtype=bool)]
    else:
        cuts = sorted(bands)
        edges = [fmin] + cuts + [fmax]
        band_labels, band_masks = [], []
        for i in range(len(edges)-1):
            lo, hi = edges[i], edges[i+1]
            mask   = Freqs <= hi if i == 0 else (Freqs > lo) & (Freqs <= hi)
            band_masks.append(mask)
            band_labels.append(f"{lo:.4g}-{hi:.4g} Hz")
        
    n_bands = len(band_labels)
    if n_bands > len(colors): print("plot_polar(): too many bands, colors will be repeated")
    colors = [colors[i % len(colors)] for i in range(n_bands)]


    # Setup plot
    if ax_in is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
    else:
        ax = ax_in
        fig = ax.figure
        
    distances = np.full(N_obs, r)
    theta_plot = np.append(theta_rad, theta_rad[0])

    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)

    mode = mode.upper()
    
    if mode == "OASPL":

        all_oaspl_vals = []
        for label, mask, color in zip(band_labels, band_masks, colors):
            oaspl = au.OASPL_band(p, mask, p_ref, absorption, Freqs, distances)
            oaspl_plot = np.append(oaspl, oaspl[0])
            all_oaspl_vals.append(oaspl)

            # Plot
            ax.plot(theta_plot, oaspl_plot, lw=2, color=color, label=label)
            ax.fill(theta_plot, oaspl_plot, alpha=0.08, color=color)

        if all_oaspl_vals:
            vals = np.concatenate(all_oaspl_vals)
            rmin, rmax = vals.min(), vals.max()
            margin = abs(rmax-rmin) * 0.1 if rmax != rmin else abs(rmin) * 0.1
            if margin == 0: margin = 1.0
            ax.set_rmin(rmin - margin)

    else:
        # Find closest frequency index for single-frequency modes
        idx = np.argmin(np.abs(Freqs - target_frequency))
        f   = Freqs[idx]
        p = p[idx, :]

        vals, unit_label = pu.convert_to(p, mode)

        if absorption and mode == "SPL": vals = au.add_absorption(f, vals, distances)

        vals_plot = np.append(vals, vals[0])
        color = colors[0] 

        # Plot
        ax.plot(theta_plot, vals_plot, lw=2, color=color, label= f"{unit_label} at {f:.2f} [Hz]")
        ax.fill(theta_plot, vals_plot, alpha=0.08, color=color)

        rmin, rmax = vals.min(), vals.max()
        margin = abs(rmax-rmin) * 0.1 if rmax != rmin else abs(rmin) * 0.1
        if margin == 0: margin = 1.0 # Fallback 
        ax.set_rmin(rmin - margin)

    ax.set_title(f"{tag}")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    
    if ax_in is None:
        fig.tight_layout(rect=[0, 0.06, 1, 1])

    return fig, ax


# ---------- Cylindrical Plots ---------- #
def plot_cylinder(case            : dict = None,        # [-] Dictionary containing simulation data
                  mode            : str = 'OASPL',      # [-] Magnitude to plot: 'OASPL', 'SPL', 'SWL', 'ABS', 'REAL', 'IMAG', 'PHASE'
                  target_frequency: float = 0.88,       # [Hz] Target frequency for single-frequency modes
                  absorption      : bool = True,        # [-] Whether to apply absorption attenuation
                  filter_under    : float = None,       # [Hz] Lower frequency cutoff (f >= filter_under)
                  filter_over     : float = None,       # [Hz] Upper frequency cutoff (f <= filter_over)
                  figsize         : tuple = (12,5),     # [-] Figure size
                  cmap            : str = 'inferno',    # [-] Contour colormap
                  ax_in           : plt.Axes = None):   # [-] Optional external axes for subplots
    """
    Plots 2D unwrapped cylindrical surface maps for acoustic magnitudes (OASPL, SPL, SWL, etc.).

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes (or array of Axes for multi-band OASPL)
    """

    if not case["has_cylinder"]:
        print("plot_cylinder(): no cylinder data, skipping")
        return None, None

    tag = case["Case_type"]

    # Extract data
    theta     = case["Theta_deg_cylinder"]   # 1D array of angles [deg]
    z         = case["Z_cylinder"]           # 1D array of height/depth [m]
    p         = case["P_cylinder"]           # Pressure grid
    Freqs     = case["Freqs"]
    p_ref     = case["p_ref"]
    r         = case["R_cylinder"]
    observers = case["Obs_cylinder"]
    cx, cy    = case["Center_cylinder"]

    nf, ntheta, nz = len(Freqs), len(theta), len(z)

    # Reshape
    observers = observers.reshape((ntheta*nz,3))
    p         = p.reshape((nf, ntheta*nz))

    # Distance grid for absorption
    distances = np.full(ntheta*nz, r, dtype=float)

    # Meshgrid and differential of area
    if "dA_cylinder" in case:
        dA = case["dA_cylinder"]
    else:
        dtheta = 2*np.pi / ntheta
        dz     = (z.max() - z.min())/(nz-1) if nz > 1 else 1.0
        dA     = r * dz * dtheta   

    THETA, Z = np.meshgrid(theta, z)

    # Filter frequency
    freq_mask = np.ones(nf, dtype=bool)
    if filter_under is not None:
        freq_mask &= (Freqs >= filter_under)
    if filter_over is not None:
        freq_mask &= (Freqs <= filter_over)
    if not np.any(freq_mask):
        print("plot_cylinder(): no frequencies match the specified filter bounds!")
        return None, None

    p, Freqs = p[freq_mask], Freqs[freq_mask]
    nf = len(Freqs)

    mode = mode.upper()
    
    if ax_in is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax_in
        fig = ax.figure

    title_extra = ""
    # Select mode
    if mode == 'OASPL':
        vals = au.pressure_to_OASPL(p, p_ref, absorption, Freqs, distances)
        unit_label = r"OASPL [dB re 1$\mu$ Pa]"

    elif mode == 'SWL':
        print("plot_cylinder(): SWL not yet implemented")
    else:
        idx = np.argmin(np.abs(Freqs - target_frequency))
        f = Freqs[idx]
        p = p[idx,:]
        title_extra = f"at {f:.2f} Hz"

        vals, unit_label = pu.convert_to(p, mode)

        if absorption and mode =='SPL': vals = au.add_absorption(f, vals, distances)

    # Plot
    theta_plot = np.append(theta, 360.)
    vals = vals.reshape((ntheta, nz))
    vals_plot = np.append(vals, vals[0:1, :], axis=0)

    # Matplotlib contour expects Z shape (len(y), len(x)), so transpose vals_plot
    contour_set = ax.contourf(theta_plot, z, vals_plot.T, levels=60, cmap=cmap)
    ax.contour(theta_plot, z, vals_plot.T, levels=30, colors='k', linewidths=0.4, alpha=0.5)
    cbar = fig.colorbar(contour_set, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(unit_label)

    ax.set_xlabel(r'Azimuth Angle, $\theta$ [deg]')
    ax.set_ylabel('Depth, z [m]')
    ax.set_title(f"{tag} - {mode} {title_extra}")
    ax.set_xlim(theta_plot.min(), theta_plot.max())
    ax.set_ylim(z.min(), z.max())
    ax.set_xticks(np.arange(0, 361, 45))

    if ax_in is None:
        fig.tight_layout()
        
    return fig, ax


# ---------- Linear Plots ---------- #
def plot_distance_decay(case            : dict  = None,         # [-] Dictionary containing simulation data
                        mode            : str   = 'OASPL',      # [-] Magnitude to plot: 'OASPL', 'SPL', 'SWL', 'ABS', 'REAL', 'IMAG', 'PHASE'
                        target_frequency: float = 0.88,         # [Hz] Target frequency for single-frequency modes
                        absorption      : bool  = True,         # [-] Whether to apply absorption attenuation
                        filter_under    : float = None,         # [Hz] Lower frequency cutoff (f >= filter_under)
                        filter_over     : float = None,         # [Hz] Upper frequency cutoff (f <= filter_over)
                        turbine_data    : bool  = False,        # [-] Wheter to plot additional turbine data from GE 2025 
                        figsize         : tuple = (10,5),       # [-] Figure size
                        ax_in           : plt.Axes = None):     # [-] Optional external axes for subplots
    """
    Plots acoustic magnitude decay over distance (1D line plot).

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
    """

    if not case["has_decay"]:
        print("plot_distance_decay(): no decay data, skipping")
        return None, None

    tag = case["Case_type"]

    # Extract data
    p         = case["P_decay"]
    Freqs     = case["Freqs"]
    observers = case["Obs_decay"]
    r         = case["Distances_decay"]
    z         = case["Z_decay"]
    rlim      = case["Distances_decay"]
    p_ref     = case["p_ref"]
    do_log    = case["Logspace_decay"]

    Nfreqs, Nr = len(Freqs), len(r)

    # Filter frequencies
    freq_mask = np.ones(Nfreqs, dtype=bool)
    if filter_under is not None:
        freq_mask &= (Freqs >= filter_under)
    if filter_over is not None:
        freq_mask &= (Freqs <= filter_over)
    if not np.any(freq_mask):
        print("plot_distance_decay(): no frequencies match the specified filter bounds!")
        return None, None

    p, Freqs = p[freq_mask], Freqs[freq_mask]
    NFreqs = len(Freqs)

    mode = mode.upper()
    
    if ax_in is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax_in
        fig = ax.figure
        
    title_extra = ""
    color = pu.get_case_color(case)

    # Select mode
    if mode == 'OASPL':
        vals = au.pressure_to_OASPL(p, p_ref, absorption, Freqs, r)
        unit_label = r"OASPL [dB re $\mu$ Pa]"
    else:
        idx = np.argmin(np.abs(Freqs - target_frequency))
        f   = Freqs[idx]
        p   = p[idx,:]
        title_extra = f"at {f:.2f} Hz"

        vals, unit_label = pu.convert_to(p, mode)
        if absorption and mode == 'SPL': vals = au.add_absorption(f, vals, r)

    # Plot
    if do_log:
        ax.semilogx(r, vals, lw=2, color=color)
        min_val = vals.min()
        if not turbine_data or True:
            ref_vals_1 = vals[0] - 10 * np.log10(np.maximum(r, np.finfo(float).eps) / r[0])
            ref_vals_2 = vals[0] - 20 * np.log10(np.maximum(r, np.finfo(float).eps) / r[0])
            ax.semilogx(r, ref_vals_1, '--', color=color, alpha=0.4, lw=1.0, label=r"$1/r$")
            ax.semilogx(r, ref_vals_2, ':', color=color, alpha=0.4, lw=1.0, label=r"$1/r^2$")
            min_val = min(ref_vals_2.min(), vals.min())
    else:
        ax.plot(r, vals, lw=2, color=color)
        min_val = vals.min()
        if not turbine_data or True:
            ref_vals_1 = vals[0] - 10 * np.log10(np.maximum(r, np.finfo(float).eps) / r[0])
            ref_vals_2 = vals[0] - 20 * np.log10(np.maximum(r, np.finfo(float).eps) / r[0])
            ax.plot(r, ref_vals_1, '--', color=color, alpha=0.4, lw=1.0, label=r"$1/r$")
            ax.plot(r, ref_vals_2, ':', color=color, alpha=0.4, lw=1.0, label=r"$1/r^2$")
            min_val = min(ref_vals_2.min(), vals.min())

    # Plot data from Ge2025
    if turbine_data:
        TURBINE_STYLES = pu.get_tubine_styles()
        all_turbine_x, all_turbine_y = [], []
        for (data, label, marker, color) in TURBINE_STYLES:
            ax.scatter(data[:, 0], data[:, 1],
                    marker=marker, color=color, s=40,
                    alpha=1.0, zorder=5, label=label, edgecolors='k', linewidths=0.3)

            all_turbine_x.append(data[:,0])
            all_turbine_y.append(data[:,1])
            min_val = min(min_val, data[:,1].min()*0.95)

        # Regresion
        tx, ty = np.concatenate(all_turbine_x), np.concatenate(all_turbine_y)

        log_tx = np.log10(tx)
        a, b   = np.polyfit(log_tx, ty, 1)
        x_fit  = np.logspace(np.log10(tx.min()), np.log10(tx.max()), len(r))
        y_fit  = a * np.log10(x_fit) + b

        ax.semilogx(x_fit, y_fit, lw=2.0, color="lightcoral", ls="--", alpha=0.5, label="Regresion -20.4 dB/decade")

    ax.grid(True, which='both', ls='--', alpha=0.7)
    ax.fill_between(r, vals, min_val, alpha=0.1, color=color)

    ax.set_xlabel("Distance, r [m]")
    ax.set_ylabel(unit_label)
    ax.set_title(f"{tag} - {mode} {title_extra}")
    ax.set_xlim(r.min(), r.max())
    ax.set_ylim(bottom=min_val)

    if ax_in is None:
        fig.tight_layout()

    return fig, ax

def plot_line(case            : dict  = None,         # [-] Dictionary containing simulation data
              mode            : str   = 'OASPL',      # [-] Magnitude to plot: 'OASPL', 'SPL', 'SWL', 'ABS', 'REAL', 'IMAG', 'PHASE'
              target_frequency: float = 0.88,         # [Hz] Target frequency for single-frequency modes
              absorption      : bool  = True,         # [-] Whether to apply absorption attenuation
              filter_under    : float = None,         # [Hz] Lower frequency cutoff (f >= filter_under)
              filter_over     : float = None,         # [Hz] Upper frequency cutoff (f <= filter_over)
              figsize         : tuple = (10,5),       # [-] Figure size
              ax_in           : plt.Axes = None):     # [-] Optional external axes for subplots
    """
    Plots acoustic magnitude along an arbitrary straight line between two points.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
    """

    if not case["has_line"]:
        print("plot_line(): no line data, skipping")
        return None, None

    tag = case["Case_type"]

    # Extract data
    p         = case["P_line"]
    Freqs     = case["Freqs"]
    r         = case["Distances_line"]  # Distances from P1 along the line
    p1        = case["P1_line"]
    p2        = case["P2_line"]
    p_ref     = case["p_ref"]
    do_log    = case["Logspace_line"]

    Nfreqs, Nr = len(Freqs), len(r)

    # Filter frequencies
    freq_mask = np.ones(Nfreqs, dtype=bool)
    if filter_under is not None:
        freq_mask &= (Freqs >= filter_under)
    if filter_over is not None:
        freq_mask &= (Freqs <= filter_over)
    if not np.any(freq_mask):
        print("plot_line(): no frequencies match the specified filter bounds!")
        return None, None

    p, Freqs = p[freq_mask], Freqs[freq_mask]
    NFreqs = len(Freqs)

    mode = mode.upper()
    
    if ax_in is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax_in
        fig = ax.figure
        
    title_extra = ""
    color = pu.get_case_color(case)

    # Select mode
    if mode == 'OASPL':
        vals = au.pressure_to_OASPL(p, p_ref, absorption, Freqs, r)
        unit_label = r"OASPL [dB re $\mu$ Pa]"
    else:
        idx = np.argmin(np.abs(Freqs - target_frequency))
        f   = Freqs[idx]
        p_f = p[idx, :]
        title_extra = f"at {f:.2f} Hz"

        vals, unit_label = pu.convert_to(p_f, mode)
        if absorption and mode == 'SPL': vals = au.add_absorption(f, vals, r)

    # Plot
    if do_log:
        ax.semilogx(r, vals, lw=2, color=color)
    else:
        ax.plot(r, vals, lw=2, color=color)

    min_val = vals.min()
    
    ax.grid(True, which='both', ls='--', alpha=0.7)
    ax.fill_between(r, vals, 0, alpha=0.1, color=color)

    # Add text box with P1 and P2 coordinates for context
    coord_text = f"P1: {np.round(p1, 1)} m\nP2: {np.round(p2, 1)} m"
    ax.text(0.97, 0.95, coord_text, transform=ax.transAxes, ha='right', va='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'),
            fontsize=9)

    ax.set_xlabel("Distance from P1 [m]")
    ax.set_ylabel(unit_label)
    ax.set_title(f"{tag} - {mode} {title_extra}")
    ax.set_xlim(r.min(), r.max())
    ax.set_ylim(vals.min(), vals.max())

    if ax_in is None:
        fig.tight_layout()

    return fig, ax

# ---------- Slice Plots ---------- #
def plot_sliceXY(case            : dict  = None,         # [-] Dictionary containing simulation data
                 mode            : str   = 'OASPL',      # [-] Magnitude to plot: 'OASPL', 'SPL', 'SWL', 'ABS', 'REAL', 'IMAG', 'PHASE'
                 target_frequency: float = 0.88,         # [Hz] Target frequency for single-frequency modes
                 absorption      : bool  = True,         # [-] Whether to apply absorption attenuation
                 filter_under    : float = None,         # [Hz] Lower frequency cutoff (f >= filter_under)
                 filter_over     : float = None,         # [Hz] Upper frequency cutoff (f <= filter_over)
                 structure       : bool = True,          # [-] Wheter to plot structure on slice
                 figsize         : tuple = (7,7),        # [-] Figure size
                 cmap            : str   = 'inferno',    # [-] Contour colormap
                 ax_in           : plt.Axes = None):     # [-] Optional external axes for subplots
    """
    Plots a 2D spatial slice in the XY plane of acoustic magnitudes.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
    """

    if not case["has_slicexy"]:
        print("plot_sliceXY(): no XY slice data, skipping")
        return None, None

    tag = case["Case_type"]

    # Extract data
    x         = case["X_slicexy"]           # 1D array of X coordinates [m]
    y         = case["Y_slicexy"]           # 1D array of Y coordinates [m]
    z_slice   = case["Z_slicexy"]           # Z level of the slice [m]
    p         = case["P_slicexy"]           # Pressure grid
    Freqs     = case["Freqs"]
    observers = case["Obs_slicexy"]
    p_ref     = case["p_ref"]
    cx, cy    = case["Center_slicexy"]

    Nfreqs, nx, ny = len(Freqs), len(x), len(y)
    distances = np.sqrt((observers[..., 0] - cx)**2 + (observers[..., 1] - cy)**2)

    # Filter frequencies
    freq_mask = np.ones(Nfreqs, dtype=bool)
    if filter_under is not None:
        freq_mask &= (Freqs >= filter_under)
    if filter_over is not None:
        freq_mask &= (Freqs <= filter_over)
    if not np.any(freq_mask):
        print("plot_sliceXY(): no frequencies match the specified filter bounds!")
        return None, None

    p, Freqs = p[freq_mask], Freqs[freq_mask]
    NFreqs = len(Freqs)

    mode = mode.upper()
    
    if ax_in is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax_in
        fig = ax.figure
        
    title_extra = ""
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Select mode
    if mode == 'OASPL':
        vals = au.pressure_to_OASPL(p, p_ref, absorption, Freqs, distances)
        unit_label = r"OASPL [dB re 1$\mu$ Pa]"
    elif mode == 'SWL':
        print("WARNING PLot_sliceXY(): SWL not yet implemented")
        pass
    else:
        idx = np.argmin(np.abs(Freqs - target_frequency))
        f   = Freqs[idx]
        p_f = p[idx, :]
        title_extra = f"at {f:.2f} Hz"

        vals, unit_label = pu.convert_to(p_f, mode)
        if absorption and mode == 'SPL': vals = au.add_absorption(f, vals, distances)

    # Plot
    contour_set = ax.contourf(X, Y, vals, levels=60, cmap=cmap)
    cbar = fig.colorbar(contour_set, ax=ax, orientation='horizontal', pad = 0.1)
    cbar.set_label(unit_label)
    if abs(x[-1]-x[0]) == abs(y[-1]-y[0]): ax.set_aspect('equal')

    # Structure
    if structure:
        try:
            nodes = case["Structure_nodes"]
            for member in nodes:
                ax.plot(member[:,0], member[:,1], '-o', color='black', markersize=2, linewidth=1.0, zorder=10)
        except Exception as e:
            print(f"plot_sliceXY() WARNING: Could not plot structure. ({e})")


    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f"{tag} - {mode} {title_extra} - Depth = {z_slice:.1f} [m]")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    if ax_in is None:
        fig.tight_layout()

    return fig, ax

def plot_sliceXZ(case            : dict  = None,         # [-] Dictionary containing simulation data
                 mode            : str   = 'OASPL',      # [-] Magnitude to plot: 'OASPL', 'SPL', 'SWL', 'ABS', 'REAL', 'IMAG', 'PHASE'
                 target_frequency: float = 0.88,         # [Hz] Target frequency for single-frequency modes
                 absorption      : bool  = True,         # [-] Whether to apply absorption attenuation
                 filter_under    : float = None,         # [Hz] Lower frequency cutoff (f >= filter_under)
                 filter_over     : float = None,         # [Hz] Upper frequency cutoff (f <= filter_over)
                 structure       : bool  = True,         # [-] Wheter to plot structure on slice
                 figsize         : tuple = (7,7),        # [-] Figure size
                 cmap            : str   = 'inferno',    # [-] Contour colormap
                 ax_in           : plt.Axes = None):     # [-] Optional external axes for subplots
    """
    Plots a 2D spatial slice in the XZ vertical plane of acoustic magnitudes.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
    """

    if not case["has_slicexz"]:
        print("plot_sliceXZ(): no XZ slice data, skipping")
        return None, None

    tag = case["Case_type"]

    # Extract data
    x         = case["X_slicexz"]           # 1D array of X coordinates [m]
    z         = case["Z_slicexz"]           # 1D array of Z coordinates [m]
    y_slice   = case["Y_slicexz"]           # Y level of the slice [m]
    p         = case["P_slicexz"]           # Pressure grid (Nf, nx, nz)
    Freqs     = case["Freqs"]
    observers = case["Obs_slicexz"]         # Observer grid (nx, nz, 3)
    p_ref     = case["p_ref"]
    
    cx, cz = np.mean(x), np.mean(z)
    Nfreqs, nx, nz = len(Freqs), len(x), len(z)
    distances = np.sqrt((observers[...,0]-cx)**2 + (observers[...,2]-cz)**2)

    # Filter frequencies
    freq_mask = np.ones(Nfreqs, dtype=bool)
    if filter_under is not None:
        freq_mask &= (Freqs >= filter_under)
    if filter_over is not None:
        freq_mask &= (Freqs <= filter_over)
    if not np.any(freq_mask):
        print("plot_sliceXZ(): no frequencies match the specified filter bounds!")
        return None, None

    p, Freqs = p[freq_mask], Freqs[freq_mask]
    NFreqs = len(Freqs)

    mode = mode.upper()
    
    if ax_in is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax_in
        fig = ax.figure
        
    title_extra = ""
    X, Z = np.meshgrid(x, z, indexing='ij')

    # Select mode
    if mode == 'OASPL':
        vals = au.pressure_to_OASPL(p, p_ref, absorption, Freqs, distances)
        unit_label = r"OASPL [dB re 1$\mu$ Pa]"
    elif mode == 'SWL':
        print("WARNING plot_sliceXZ(): SWL not yet implemented")
        pass
    else:
        idx = np.argmin(np.abs(Freqs - target_frequency))
        f   = Freqs[idx]
        p_f = p[idx, :, :]
        title_extra = f"at {f:.2f} Hz"

        vals, unit_label = pu.convert_to(p_f, mode)
        if absorption and mode == 'SPL': vals = au.add_absorption(f, vals, distances)

    # Plot
    contour_set = ax.contourf(X, Z, vals, levels=60, cmap=cmap)
    cbar = fig.colorbar(contour_set, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label(unit_label)
    if abs(x[-1]-x[0]) == abs(z[-1]-z[0]): ax.set_aspect('equal')

    if structure:
            try:
                nodes = case["Structure_nodes"]
                for member in nodes:
                    ax.plot(member[:,0], member[:,2], '-o', color='black', markersize=2, linewidth=1.0, zorder=10)
            except Exception as e:
                print(f"plot_sliceXY() WARNING: Could not plot structure. ({e})")

    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title(f"{tag} - {mode} {title_extra} - Y = {y_slice:.1f} [m]")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(z.min(), nodes[:,:,2].max() if structure else z.max())

    if ax_in is None:
        fig.tight_layout()

    return fig, ax

# ---------- Compute metrics ---------- #
def compute_sphere_metrics(case            : dict  = None,         # [-] Dictionary containing simulation data
                           absorption      : bool  = True,         # [-] Whether to apply absorption attenuation for OASPL
                           filter_under    : float = None,         # [Hz] Lower frequency cutoff (f >= filter_under)
                           filter_over     : float = None,         # [Hz] Upper frequency cutoff (f <= filter_over)
                           rho_water       : float = 1025.0,       # [kg/m³] Water density
                           c_water         : float = 1500.0,       # [m/s] Speed of sound in water
                           verbose         : bool  = True):        # [-] Print metrics to console
    """
    Computes and optionally prints radiation metrics from a spherical pressure field.
    
    Returns
    -------
    metrics : dict
        Dictionary containing all computed acoustic metrics (Power, SL, etc.)
    """

    if not case["has_sphere"]:
        print("compute_sphere_metrics(): no sphere data found in case, skipping.")
        return None
    
    rho_wat = case.get("rho_wat", rho_water)
    c_wat   = case.get("c_wat", c_water) 
    tag     = case["Case_type"]

    # Extract data
    p       = case["P_sphere"]
    Freqs   = case["Freqs"]
    r       = case["R_sphere"]
    dA      = case["dA_sphere"]
    p_ref   = case["p_ref"]
    n_theta = case["N_theta"]
    nz      = case["Nz_sphere"]

    A_wet = case.get("wet_area", 848.12 if "Monopile" in tag else 9359.07)
    print("WARNING compute_sphere_metrics(): Wetted Area is harcoded. OUTPUT FROM WT")

    # Filter frequencies
    Nfreqs = len(Freqs)
    freq_mask = np.ones(Nfreqs, dtype=bool)
    if filter_under is not None:
        freq_mask &= (Freqs >= filter_under)
    if filter_over is not None:
        freq_mask &= (Freqs <= filter_over) 
    if not np.any(freq_mask):
        print("compute_sphere_metrics(): no frequencies match the specified filter bounds!")
        return None

    p, Freqs  = p[freq_mask], Freqs[freq_mask]
    distances = np.full((n_theta, nz), r)

    # --- 1. Compute OASPL and Directional SL --- #
    oaspl = au.pressure_to_OASPL(p, p_ref, absorption, Freqs, distances)
    sl    = oaspl + 20 * np.log10(r)

    # --- 2. Compute Radiated Power (W) --- #
    p_sq = np.sum(np.abs(p)**2, axis=0)
    I = p_sq / (2.*rho_wat*c_wat)
    W = np.sum(I*dA)
    W_dB = 10* np.log10(W/1e-12)

    # --- 3. Compute Power per Wet Area --- #
    W_per_wet = W / A_wet
    W_per_wet_dB = 10.0 * np.log10(W_per_wet / 1e-12)

    # --- 4. Equivalent Omnidirectional SL --- #
    W_ref_pressure = (4.0 * np.pi * p_ref**2) / (rho_water * c_water)
    SL_omni = 10.0 * np.log10(W / W_ref_pressure)

    # --- 5. Directional SL Statistics --- #
    SL_max = np.max(sl)
    SL_min = np.min(sl)
    SL_mean_power = 10.0 * np.log10(np.mean(10.0**(sl / 10.0))) # Energetic mean

    # --- 6. Mean SPL on the sphere --- #
    SPL_sphere = 10.0 * np.log10(np.mean(p_sq) / p_ref**2)

    # --- Package results --- #
    metrics = {
        "OASPL_sphere": oaspl,
        "SL_sphere": sl,
        "W": W,
        "W_dB": W_dB,
        "W_per_wet": W_per_wet,
        "W_per_wet_dB": W_per_wet_dB,
        "SL_omni": SL_omni,
        "SL_max": SL_max,
        "SL_min": SL_min,
        "SL_mean_power": SL_mean_power,
        "SPL_sphere_mean": SPL_sphere
    }

    # --- Print formatting --- #
    if verbose:
        print("\n" + "=" * 80)
        print(f"  RADIATION METRICS (SPHERE r = {r:.0f} m, free-field)")
        
        if filter_under is not None or filter_over is not None:
            lo = f"{filter_under:.1f}" if filter_under is not None else "0"
            hi = f"{filter_over:.1f}"  if filter_over is not None else "∞"
            print(f"  Frequency band: {lo} - {hi} Hz")
        print("=" * 80)

        label_w = 42
        print(f"\n[Case: {tag}]")
        print(f"  {'Radius of sphere':<{label_w}}: {r:.1f} m")
        print(f"  {'Wet area of structure':<{label_w}}: {A_wet:.1f} m²")
        print(f"  {'Total radiated power (W)':<{label_w}}: {W:.4e} W")
        print(f"  {'Total radiated power (dB)':<{label_w}}: {W_dB:.1f} dB re 1 pW")
        print(f"  {'Power / wet area (W)':<{label_w}}: {W_per_wet:.4e} W/m²")
        print(f"  {'Power / wet area (dB)':<{label_w}}: {W_per_wet_dB:.1f} dB re 1 pW/m²")
        print(f"  {'Equivalent omnidirectional SL':<{label_w}}: {SL_omni:.1f} dB re 1 µPa @ 1 m")
        print(f"  {'Directional SL MAX':<{label_w}}: {SL_max:.1f} dB re 1 µPa @ 1 m")
        print(f"  {'Directional SL MIN':<{label_w}}: {SL_min:.1f} dB re 1 µPa @ 1 m")
        print(f"  {'Directional SL mean (energ.)':<{label_w}}: {SL_mean_power:.1f} dB re 1 µPa @ 1 m")
        print(f"  {'Mean SPL on sphere':<{label_w}}: {SPL_sphere:.1f} dB re 1 µPa")

    return metrics

def compute_cylinder_metrics(case            : dict  = None,         # [-] Dictionary containing simulation data
                             filter_under    : float = None,         # [Hz] Lower frequency cutoff (f >= filter_under)
                             filter_over     : float = None,         # [Hz] Upper frequency cutoff (f <= filter_over)
                             rho_water       : float = 1025.0,       # [kg/m³] Water density
                             c_water         : float = 1500.0,       # [m/s] Speed of sound in water
                             verbose         : bool  = True):        # [-] Print metrics to console
    """
    Computes and optionally prints propagation metrics from a cylindrical pressure field.
    
    Returns
    -------
    metrics : dict
        Dictionary containing all computed acoustic metrics (Power, SPL, TL, etc.)
    """

    if not case["has_cylinder"]:
        print("plot_cylinder(): no cylinder data, skipping")
        return None, None
    
    rho_wat = case.get("rho_wat", rho_water)
    c_wat   = case.get("c_wat", c_water) 
    tag     = case["Case_type"]

    # Extract data
    p       = case["P_cylinder"]
    Freqs   = case["Freqs"]
    r       = case["R_cylinder"]
    z       = case["Z_cylinder"]
    theta   = case["Theta_deg_cylinder"]
    p_ref   = case["p_ref"]
    
    n_theta, nz = len(theta), len(z)

    # Ensure theta is in radians for integration
    theta_rad = theta if np.max(theta) <= 2*np.pi + 0.1 else np.deg2rad(theta)

    # Filter frequencies
    Nfreqs = len(Freqs)
    freq_mask = np.ones(Nfreqs, dtype=bool)
    if filter_under is not None:
        freq_mask &= (Freqs >= filter_under)
    if filter_over is not None:
        freq_mask &= (Freqs <= filter_over) 
    if not np.any(freq_mask):
        print("compute_cylinder_metrics(): no frequencies match the specified filter bounds!")
        return None

    p, Freqs = p[freq_mask], Freqs[freq_mask]

    # --- 1. Intensity and Integration Setup --- #
    p_sq = np.sum(np.abs(p)**2, axis=0)      # Sum energy over frequencies: shape (n_theta, nz)
    I = p_sq / (2.0 * rho_wat * c_wat)

    A = 2 * np.pi * r * np.max(np.abs(z))    # Lateral area of the cylinder

    # --- 2. Compute Radiated Power (W) --- #
    I_vs_z = np.trapz(I * r, theta_rad, axis=0)
    W = np.abs(np.trapz(I_vs_z, z))          # Absolute to handle z-axis integration direction safely
    W_dB = 10 * np.log10(W / 1e-12)

    # --- 3. Compute Power per Cylinder Area --- #
    W_per_area = W / A
    W_per_area_dB = 10 * np.log10(W_per_area / 1e-12)

    # --- 4. Mean SPL on the cylinder --- #
    p2_avg = np.mean(p_sq)
    SPL_avg = 10 * np.log10(p2_avg / p_ref**2)

    # --- 5. Apparent Omnidirectional SL (from Far Field) --- #
    W_ref_press = (4.0 * np.pi * p_ref**2) / (rho_wat * c_wat)
    SL_omni_far = 10.0 * np.log10(W / W_ref_press)

    # --- 6. Comparative Metrics (if Sphere data exists) --- #
    TL = None
    power_loss = None
    TL_str = ""
    power_loss_str = ""

    # --- Package results ---
    metrics = {
        "W_cyl": W,
        "W_cyl_dB": W_dB,
        "W_cyl_per_area": W_per_area,
        "W_cyl_per_area_dB": W_per_area_dB,
        "SPL_cyl_mean": SPL_avg,
        "SL_omni_far": SL_omni_far,
        "TL_cyl": TL,
        "Power_loss_cyl": power_loss
    }

    # --- Print formatting ---
    if verbose:
        print("\n" + "=" * 80)
        print(f"  PROPAGATION METRICS (CYLINDER r = {r:.0f} m)")
        
        if filter_under is not None or filter_over is not None:
            lo = f"{filter_under:.1f}" if filter_under is not None else "0"
            hi = f"{filter_over:.1f}"  if filter_over is not None else "∞"
            print(f"  Frequency band: {lo} - {hi} Hz")
        print("=" * 80)

        label_w = 44
        print(f"\n[Case: {tag}]")
        print(f"  {'Radius of cylinder':<{label_w}}: {r:.1f} m")
        print(f"  {'Height of cylinder (H)':<{label_w}}: {np.max(np.abs(z)):.1f} m")
        print(f"  {'Lateral area of cylinder':<{label_w}}: {A:.1f} m²")
        print(f"  {'Total power flow (W)':<{label_w}}: {W:.4e} W")
        print(f"  {'Total power flow (dB)':<{label_w}}: {W_dB:.1f} dB re 1 pW")
        print(f"  {'Power / cylinder area (W)':<{label_w}}: {W_per_area:.4e} W/m²")
        print(f"  {'Power / cylinder area (dB)':<{label_w}}: {W_per_area_dB:.1f} dB re 1 pW/m²")
        print(f"  {'Mean SPL on cylinder':<{label_w}}: {SPL_avg:.1f} dB re 1 µPa")
        print(f"  {'Apparent omnid. SL (from far-field)':<{label_w}}: {SL_omni_far:.1f} dB re 1 µPa @ 1 m")
        
        if TL_str:
            print(f"  {'Transmission Loss (TL)':<{label_w}}: {TL_str}")
        if power_loss_str:
            print(f"  {'Power loss (W_sphere / W_cyl)':<{label_w}}: {power_loss_str}")

    return metrics