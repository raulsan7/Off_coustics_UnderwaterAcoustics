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
                   figsize: tuple = (8,6)):     # [-] Figure size
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

        fig = plt.figure(figsize=figsize)
        ax  =fig.add_subplot(111, projection='3d')

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
        fig, ax = plt.subplots(figsize=figsize)

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
                  figsize      : tuple = (10,5)):   # [-] Figure size
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

    print(p)

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
    fig, ax       = plt.subplots(1, 1, figsize=figsize)
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
        fig.tight_layout(rect=[0, 0.06, 1, 1])
    else:
        ax.legend()
        fig.tight_layout()

    return fig, ax


# ---------- Polar Plots ---------- #
def plot_polar(case             : dict  = None,        # [-] Dictionary containing simulation data
               mode             : str   = 'OASPL',     # [-] Magnitude to plot: 'OASPL', 'SPL', or 'ABS'
               target_frequency : float = 0.88,        # [Hz] Target frequency for 'SPL' or 'ABS' modes
               absorption       : bool  = True,        # [-] Whether to apply absorption attenuation
               bands            : list  = [10., 100.], # [Hz] Frequency band limits for OASPL
               figsize          : tuple = (7,7)):      # [-] Figure size
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
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
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
                  cmap            : str = 'inferno'):   # [-] COntour colormap
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
    fig, ax = plt.subplots(figsize=figsize)

    # Select mode
    if mode == 'OASPL':
        vals = au.pressure_to_OASPL(p, p_ref, absorption, Freqs, distances)
        unit_label = r"OASPL [dB re 1$\mu$ Pa]"

    elif mode == 'SWL':
        pass
    else:
        idx = np.argmin(np.abs(Freqs - target_frequency))
        f = Freqs[idx]
        p = p[idx,:]

        vals, unit_label = pu.convert_to(p, mode)

        if absorption and mode =='SPL': vals = au.add_absorption(f, vals, distances)

    # Plot
    theta_plot = np.append(theta, 360.)
    vals = vals.reshape((ntheta, nz))
    vals_plot = np.append(vals, vals[0:1, :], axis=0)

    # Matplotlib contour expects Z shape (len(y), len(x)), so transpose vals_plot
    contour_set = ax.contourf(theta_plot, z, vals_plot.T, levels=60, cmap=cmap)
    cbar = fig.colorbar(contour_set, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(unit_label)

    ax.set_xlabel('Theta [deg]')
    ax.set_ylabel('Height / Depth [m]')
    ax.set_title(f"{tag} - {mode}")
    ax.set_xlim(theta_plot.min(), theta_plot.max())
    ax.set_ylim(z.min(), z.max())
    ax.set_xticks(np.arange(0, 361, 45))

    fig.tight_layout()
    return fig, ax


        



       



