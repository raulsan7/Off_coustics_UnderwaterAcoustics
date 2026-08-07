from turtle import clear

import numpy as np
import matplotlib.pyplot as plt
import plotting.plot_utils as pu
import utils.AcousticUtils as au
import utils.EnvironmentalData as ed



# ---------- Structural Plots ---------- #
def plot_structure(case   : dict = None,        # [-] Dictionary containing simulation data
                   mode   : str = '3D',         # [-] Plot mode for the structure representation
                   WindDir: bool = True,        # [-] Wheter to plot or not wind direction
                   figsize: tuple = (8,6),      # [-] Figure size
                   ax_in  : plt.Axes = None):   # [-] optional external axes for subplots
    """
    Plots the structure nodes in shape(Nmembers, Nnodes, 3) per turbine in the farm

    Returns
    -------
    fig, ax: matplotlib.figure.Figure, matplotlib.axes.Axes
    The generated figure and axes objet for further editing.
    """

    if not "Structure_nodes" in case["Turbine_parameters"][0]:
        print("plot_structure(): no structure data, skipping")
        return None, None

    if mode not in ("3D", "xy", "xz"):
        raise ValueError("plot_structure(): mode must be '3D', 'xy', 'xz'")

    # Extract nodes and depth
    Nturb = case["Num_turbines"]
    nodes = []
    nodes = np.asarray([turb["Structure_nodes"] for turb in case["Turbine_parameters"]])       # shape(Nturb, Nmembers, Nnodes, 3)
    Depth = case["Depth"]
    tags  = np.asarray([turb["Case_type"] for turb in case["Turbine_parameters"]])
    Wind_dir = np.asarray([turb["WindDir"] for turb in case["Turbine_parameters"]])

    # Find node with maximum Z coordinate to place WindDir arrow
    max_z_nodes = np.zeros((Nturb, 3))
    min_z_nodes = np.zeros((Nturb, 3))
    for i in range(Nturb):
        xyz = nodes[i].reshape(-1,3)
        max_z_nodes[i] = xyz[np.argmax(xyz[:,2]),:]
        min_z_nodes[i] = xyz[np.argmin(xyz[:,2]),:]

    max_x, max_y, max_z = max_z_nodes[:,0], max_z_nodes[:,1], max_z_nodes[:,2]
    min_x, min_y, min_z = min_z_nodes[:,0], min_z_nodes[:,1], min_z_nodes[:,2]


    if mode == "3D":

        # Handle axes
        if ax_in is None:
                fig = plt.figure(figsize=figsize)
                ax  = fig.add_subplot(111, projection="3d")
        else:
            ax = ax_in
            fig = ax.figure

        # Compute extents for drawing planes
        xs = nodes[...,0].ravel()
        ys = nodes[...,1].ravel()
        zs = nodes[...,2].ravel()

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        zmin, zmax = zs.min(), zs.max()
        xpad = 0.1 * (xmax-xmin) if xmax>xmin else 1.0
        ypad = 0.1 * (ymax-ymin) if ymax>ymin else 1.0
        zpad = 0.1 * (zmax-zmin) if zmax>zmin else 1.0

        # Draw seabed and surface planes
        Xg = np.linspace(xmin-xpad, xmax+xpad, 2)
        Yg = np.linspace(ymin-ypad, ymax+ypad, 2)
        XX, YY = np.meshgrid(Xg, Yg, indexing='ij')

        ax.plot_surface(XX, YY, np.full_like(XX, -Depth), color='gray', alpha=0.5, shade=True, label="Seabed")
        ax.plot_surface(XX, YY, np.full_like(XX, 0.0), color="deepskyblue", alpha=0.25, shade=False, label="Surface")

        # Plot each member as connected nodes for each turbine
        for i in range(Nturb):
            turb = nodes[i]
            for member in turb:
                ax.plot(member[:,0], member[:,1], member[:,2], "-o", color="k", ms=3)

        # Draw wind direction arrow
        if WindDir:
            scale_length = np.mean([xmax - xmin, ymax - ymin, zmax - zmin])
            if scale_length <= 0:
                scale_length = 1.0
            arrow_length = scale_length * 0.5
            if arrow_length == 0:
                arrow_length = 0.75

            for i in range(Nturb):
                pu.draw_direction_arrow(ax, max_x[i], max_y[i], Wind_dir[i],
                                        length=arrow_length, z=max_z[i], color = "red",
                                        label=f"WindDir {Wind_dir[i]}")

        x_center = 0.5 * (xmin + xmax)
        y_center = 0.5 * (ymin + ymax)

        x_span = (xmax - xmin) + 2*xpad
        y_span = (ymax - ymin) + 2*ypad
        span_xy = max(x_span, y_span)

        x_limits = (x_center - 0.5*span_xy, x_center + 0.5*span_xy)
        y_limits = (y_center - 0.5*span_xy, y_center + 0.5*span_xy)

        z_span = (zmax - zmin) + 2*zpad
        z_limits = (zmin - zpad, zmax + zpad)

        ax.legend()
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(f"3D View (XY)")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_zlim(*z_limits)
        ax.set_box_aspect((span_xy, span_xy, z_span))

    else:

        # Handle axes
        if ax_in is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = ax_in
            fig = ax.figure

        if mode == "xy":

            # Compute extents for drawing planes
            xs = nodes[..., 0].ravel()
            ys = nodes[..., 1].ravel()

            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            xpad = 0.1*(xmax-xmin) if xmax > xmin else 1.0
            ypad = 0.1*(ymax-ymin) if ymax > ymin else 1.0

            x_center = 0.5 * (xmin + xmax)
            y_center = 0.5 * (ymin + ymax)
            x_span = (xmax - xmin) + 2*xpad
            y_span = (ymax - ymin) + 2*ypad
            span_xy = max(x_span, y_span)

            x_limits = (x_center - 0.5*span_xy, x_center + 0.5*span_xy)
            y_limits = (y_center - 0.5*span_xy, y_center + 0.5*span_xy)

            # Project onto z = 0 plane
            for i in range(Nturb):
                turb = nodes[i]
                for member in turb:
                    ax.plot(member[:,0], member[:,1], '-o', color='k', ms=3)

            # Draw wind direction arrow
            if WindDir:
                for i in range(Nturb):
                    arrow_length = (xmax - xmin + 2*xpad) * 0.25
                    if arrow_length == 0: arrow_length = .75

                    pu.draw_direction_arrow(ax, max_x[i], max_y[i], Wind_dir[i],
                                            length=arrow_length, z=max_z[i], color = "red",
                                            label=f"WindDir {Wind_dir[i]}")

            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_title(f"Top View (XY)")
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
            ax.set_aspect('equal', adjustable='box')
            ax.legend()

        elif mode == 'xz':

            # Compute extents for drawing plane
            xs = nodes[..., 0].ravel()
            zs = nodes[..., 2].ravel()

            xmin, xmax = xs.min(), xs.max()
            zmin, zmax = zs.min(), zs.max()
            xpad = 0.1*(xmax-xmin) if xmax > xmin else 1.0
            zpad = 0.1*(zmax-zmin) if zmax > zmin else 1.0

            ax.axhline(0.0, color="deepskyblue", linestyle='--', linewidth=1.0, alpha=0.6, label='Surface')
            ax.axhline(-Depth, color='gray', linestyle='--', linewidth=1.0, alpha=0.6, label='Seabed')

            x_fill = np.array([xmin - xpad, xmax + xpad])
            ax.fill_between(x_fill, np.full_like(x_fill, -Depth), np.full_like(x_fill, 0.0), color='deepskyblue', alpha=0.25)

            # Project onto y = 0 plane
            for i in range(Nturb):
                turb = nodes[i]
                for member in turb:
                    ax.plot(member[:,0], member[:,2], '-o', color='k', ms=3)

            # Draw wind direction arrow (Not visible in this plane)
            if WindDir:
                arrow_length = (xmax - xmin + 2*xpad) * 0.25
                if arrow_length == 0: arrow_length = .75
                if Wind_dir[i] == 0.0 or Wind_dir[i] == 180.0:
                    pu.draw_direction_arrow(ax, max_x[i], max_y[i], Wind_dir[i],
                                        length=arrow_length, color = "red",
                                        label=f"WindDir {Wind_dir[i]}")

            ax.set_xlabel('x [m]')
            ax.set_ylabel('z [m]')
            ax.set_title(f"Side View (XZ)")
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(zmin - zpad, zmax + zpad)
            ax.legend()     

    return fig, ax


# ---------- Spectrum Plots ---------- #
def plot_spectrums(case         : dict     = None,      # [-] Dictionary containing simulation data
                   mode         : str      = 'SPL',     # [-] Selects Sound Pressure/Power Level 'SPL' or 'SWL'
                   absorption   : bool     = False,     # [-] Wheter to apply approximate absorption attenuation
                   octave       : bool     = True,      # [-] 1/3 octave or fine resolution
                   filter_under : float    = None,      # [Hz] Minimum frequency threshold
                   filter_over  : float    = None,      # [Hz] Maximum frequency threshold
                   do_thresholds: bool     = False,     # [-] Whether to plot regulatory/reference threholds
                   figsize      : tuple    = (10,5),    # [-] Figure size
                   ax_in        : plt.Axes = None):     # [-] Optional external axes for subplots
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
    if absorption:
        print("plot_spectrums(): distances non defined since there are more than one turbine: Setting absorption = False")

    tags = np.asarray([turb["Case_type"] for turb in case["Turbine_parameters"]])
    colors = pu.get_case_color(tags=tags); color = colors[0]
    Nturb = case["Num_turbines"]

    # Extract data
    Freqs     = case["Freqs"]
    Observers = case["Obs_spectrums"]
    p         = case["P_spectrums"]
    AxisPos   = np.asarray([turb["AxisPos"] for turb in case["Turbine_parameters"]])
    p_ref     = case["p_ref"]

    if mode == 'SPL':
        spl = au.pressure_to_SPL(p, p_ref)
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
    oaspl = au.pressure_to_OASPL(p[mask], p_ref)
    
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
        if Observers.ndim > 1:
            coord = tuple(float(f"{val:.1f}") for val in Observers[i])
        else:
            coord = (float(f"{Observers[i]:.1f}"),)
        ax.semilogx(Freqs, y_data, ls, color=color, label=f"{coord} [m]")
    
        print(f"\nObserver coordinates: {coord} [m]")
        print(f"OASPL = {oaspl[i]:.1f}")
        
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



    

