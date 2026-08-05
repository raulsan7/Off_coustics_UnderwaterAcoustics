import re
import matplotlib.pyplot as plt
import plotting.plot_single_case as psc

# ---------- Structural Plots ---------- #
def plot_structure(cases  : list  = None,      # [-] List of dictionaries containing simulation data
                   mode   : str   = '3D',       # [-] Plot mode for the structure representation ('3D', 'xy', 'xz')
                   WindDir: bool  = True,       # [-] Whether to plot or not wind direction
                   figsize: tuple = (8, 6)):    # [-] Size of a single panel
    """
    Plots structure nodes for three different cases in a 1x3 subplot.
    Expected order in 'cases': [case_monopile, case_floating_shallow, case_floating].
    """

    if cases is None or len(cases) != 3:
        print("plot_structure(): Exactly 3 cases are required.")
        return None, None

    total_figsize = (figsize[0] * 3, figsize[1])
    subplot_kw = {'projection': '3d'} if mode == '3D' else None
    sharey = False if mode == '3D' else True
    
    fig, axes = plt.subplots(1, 3, figsize=total_figsize, subplot_kw=subplot_kw, sharey=sharey)

    for i, case in enumerate(cases):
        psc.plot_structure(case=case,
                           mode=mode,
                           WindDir=WindDir,
                           figsize=figsize,
                           ax_in=axes[i])
        if i > 0 and mode != '3D':
            axes[i].set_ylabel("")

    fig.tight_layout()
    return fig, axes


# ---------- Spectrum Plots ---------- #
def plot_spectrum(cases        : list  = None,      # [-] List of dictionaries containing simulation data
                  mode         : str   = 'SPL',     # [-] Selects Sound Pressure/Power Level. 'SPL' or 'SWL'
                  absorption   : bool  = True,      # [-] Whether to apply approximate absorption attenuation
                  octave       : bool  = True,      # [-] 1/3 octave or fine resolution
                  filter_under : float = None,      # [Hz] Minimum frequency threshold
                  filter_over  : float = None,      # [Hz] Maximum frequency threshold
                  do_thresholds: bool  = False,     # [-] Whether to plot regulatory/reference thresholds
                  figsize      : tuple = (7, 5)):  # [-] Size of a single panel
    """
    Plots the acoustic spectrum for three different cases in a 1x3 subplot.
    Expected order in 'cases': [case_monopile, case_floating_shallow, case_floating].
    """

    if cases is None or len(cases) != 3:
        print("plot_spectrum(): Exactly 3 cases are required.")
        return None, None

    total_figsize = (figsize[0] * 3, figsize[1])
    fig, axes = plt.subplots(1, 3, figsize=total_figsize, sharey=True)

    for i, case in enumerate(cases):
        psc.plot_spectrum(case=case,
                          mode=mode,
                          absorption=absorption,
                          octave=octave,
                          filter_under=filter_under,
                          filter_over=filter_over,
                          do_thresholds=do_thresholds,
                          figsize=figsize,
                          ax_in=axes[i])
        if i > 0:
            axes[i].set_ylabel("")

    fig.tight_layout()
    return fig, axes


# ---------- Polar Plots ---------- #
def plot_polar(cases           : list  = None,         # [-] List of dictionaries containing simulation data
               mode            : str   = 'OASPL',      # [-] Magnitude to plot: 'OASPL', 'SPL', or 'ABS'
               target_frequency: float = 0.88,         # [Hz] Target frequency for 'SPL' or 'ABS' modes
               absorption      : bool  = True,         # [-] Whether to apply absorption attenuation
               bands           : list  = [10., 100.],  # [Hz] Frequency band limits for OASPL
               figsize         : tuple = (7, 7)):      # [-] Size of a single panel
    """
    Plots polar directivity diagram for three different cases in a 1x3 subplot.
    Expected order in 'cases': [case_monopile, case_floating_shallow, case_floating].
    """

    if cases is None or len(cases) != 3:
        print("plot_polar(): Exactly 3 cases are required.")
        return None, None

    total_figsize = (figsize[0] * 3, figsize[1])
    fig, axes = plt.subplots(1, 3, figsize=total_figsize, subplot_kw={"projection": "polar"})

    for i, case in enumerate(cases):
        psc.plot_polar(case=case,
                       mode=mode,
                       target_frequency=target_frequency,
                       absorption=absorption,
                       bands=bands,
                       figsize=figsize,
                       ax_in=axes[i])

    # Legend handling
    handles, labels = axes[0].get_legend_handles_labels()

    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor = (0.5, 0.02), ncol=len(handles), frameon=False)

    fig.tight_layout(rect=[0, 0.1, 1, 0.95])

    return fig, axes


# ---------- Cylindrical Plots ---------- #
def plot_cylinder(cases           : list  = None,      # [-] List of dictionaries containing simulation data
                  mode            : str   = 'OASPL',   # [-] Magnitude to plot: 'OASPL', 'SPL', 'SWL', 'ABS', 'REAL', 'IMAG', 'PHASE'
                  target_frequency: float = 0.88,      # [Hz] Target frequency for single-frequency modes
                  absorption      : bool  = True,      # [-] Whether to apply absorption attenuation
                  filter_under    : float = None,      # [Hz] Lower frequency cutoff (f >= filter_under)
                  filter_over     : float = None,      # [Hz] Upper frequency cutoff (f <= filter_over)
                  figsize         : tuple = (10, 5),   # [-] Size of a single panel
                  cmap            : str   = 'inferno'):# [-] Contour colormap
    """
    Plots 2D unwrapped cylindrical surface maps for three different cases in a 3x1 subplot.
    Expected order in 'cases': [case_monopile, case_floating_shallow, case_floating].
    """

    if cases is None or len(cases) != 3:
        print("plot_cylinder(): Exactly 3 cases are required.")
        return None, None

    total_figsize = (figsize[0], figsize[1]*3)
    fig, axes = plt.subplots(3, 1, figsize=total_figsize, sharex=True)

    case_names = []
    mode_suffix = ""

    for i, case in enumerate(cases):
        psc.plot_cylinder(case=case,
                          mode=mode,
                          target_frequency=target_frequency,
                          absorption=absorption,
                          filter_under=filter_under,
                          filter_over=filter_over,
                          figsize=figsize,
                          cmap=cmap,
                          ax_in=axes[i])
        
        # Hide x lables
        if i < 2:
            axes[i].set_xlabel("")

        # suptitle handling
        title_text = axes[i].get_title()
        axes[i].set_title("")
        
        # Separate name from magnitude label
        if " - " in title_text:
            name, suffix = title_text.rsplit(" - ", 1)
            case_names.append(name)
            # Save first suffix
            if i == 0:
                mode_suffix = suffix
        else:
            case_names.append(title_text)

    # Reconstruct gloabl title
    if mode_suffix:
        global_title = f"{', '.join(case_names)} - {mode_suffix}"
    else:
        global_title = ", ".join(case_names)

    fig.suptitle(global_title, fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, axes


# ---------- Linear Plots ---------- #
def plot_distance_decay(cases           : list  = None,      # [-] List of dictionaries containing simulation data
                        mode            : str   = 'OASPL',   # [-] Magnitude to plot: 'OASPL', 'SPL', 'SWL', 'ABS', 'REAL', 'IMAG', 'PHASE'
                        target_frequency: float = 0.88,      # [Hz] Target frequency for single-frequency modes
                        absorption      : bool  = True,      # [-] Whether to apply absorption attenuation
                        filter_under    : float = None,      # [Hz] Lower frequency cutoff (f >= filter_under)
                        filter_over     : float = None,      # [Hz] Upper frequency cutoff (f <= filter_over)
                        turbine_data    : bool  = False,     # [-] Whether to plot additional turbine data from GE 2025
                        figsize         : tuple = (7, 5)):  # [-] Size of a single panel
    """
    Plots acoustic magnitude decay over distance for three different cases in a 1x3 subplot.
    Expected order in 'cases': [case_monopile, case_floating_shallow, case_floating].
    """
    if cases is None or len(cases) != 3:
        print("plot_distance_decay(): Exactly 3 cases are required.")
        return None, None

    total_figsize = (figsize[0] * 3, figsize[1])
    fig, axes = plt.subplots(1, 3, figsize=total_figsize, sharey=True)

    for i, case in enumerate(cases):
        psc.plot_distance_decay(case=case,
                                mode=mode,
                                target_frequency=target_frequency,
                                absorption=absorption,
                                filter_under=filter_under,
                                filter_over=filter_over,
                                turbine_data=turbine_data,
                                figsize=figsize,
                                ax_in=axes[i])
        if i > 0:
            axes[i].set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    if handles:
        fig.legend(handles, labels, 
                   loc='lower center', 
                   bbox_to_anchor=(0.5, 0.02), 
                   ncol=len(handles), 
                   fontsize=12, 
                   frameon=False)
        
        fig.tight_layout(rect=[0, 0.12, 1, 1])
    else:
        fig.tight_layout()
    
    return fig, axes

def plot_line(cases           : list  = None,      # [-] List of dictionaries containing simulation data
              mode            : str   = 'OASPL',   # [-] Magnitude to plot: 'OASPL', 'SPL', 'SWL', 'ABS', 'REAL', 'IMAG', 'PHASE'
              target_frequency: float = 0.88,      # [Hz] Target frequency for single-frequency modes
              absorption      : bool  = True,      # [-] Whether to apply absorption attenuation
              filter_under    : float = None,      # [Hz] Lower frequency cutoff (f >= filter_under)
              filter_over     : float = None,      # [Hz] Upper frequency cutoff (f <= filter_over)
              figsize         : tuple = (7, 5)):  # [-] Size of a single panel
    """
    Plots acoustic magnitude along a straight line for three different cases in a 1x3 subplot.
    Expected order in 'cases': [case_monopile, case_floating_shallow, case_floating].
    """
    if cases is None or len(cases) != 3:
        print("plot_line(): Exactly 3 cases are required.")
        return None, None

    total_figsize = (figsize[0] * 3, figsize[1])
    fig, axes = plt.subplots(1, 3, figsize=total_figsize, sharey=True)

    ymin, ymax = float('inf'), float('-inf')
    for i, case in enumerate(cases):
        psc.plot_line(case=case,
                      mode=mode,
                      target_frequency=target_frequency,
                      absorption=absorption,
                      filter_under=filter_under,
                      filter_over=filter_over,
                      figsize=figsize,
                      ax_in=axes[i])
        if i > 0:
            axes[i].set_ylabel("")

        ymin_local, ymax_local = axes[i].get_ylim()

        ymin = min(ymin, ymin_local)
        ymax = max(ymax, ymax_local)

        axes[i].set_ylim(ymin, ymax)

    return fig, axes


# ---------- XY Slice Plots ---------- #
def plot_sliceXY(cases           : list  = None,      # [-] List of dictionaries containing simulation data
                 mode            : str   = 'OASPL',   # [-] Magnitude to plot: 'OASPL', 'SPL', 'SWL', 'ABS', 'REAL', 'IMAG', 'PHASE'
                 target_frequency: float = 0.88,      # [Hz] Target frequency for single-frequency modes
                 absorption      : bool  = True,      # [-] Whether to apply absorption attenuation
                 filter_under    : float = None,      # [Hz] Lower frequency cutoff (f >= filter_under)
                 filter_over     : float = None,      # [Hz] Upper frequency cutoff (f <= filter_over)
                 structure       : bool  = True,      # [-] Whether to plot structure on slice
                 figsize         : tuple = (7, 7),    # [-] Size of a single panel
                 cmap            : str   = 'inferno'):# [-] Contour colormap
    """
    Plots a 2D spatial XY slice for three different cases in a 1x3 subplot.
    Expected order in 'cases': [case_monopile, case_floating_shallow, case_floating].
    """

    if cases is None or len(cases) != 3:
        print("plot_sliceXY(): Exactly 3 cases are required.")
        return None, None

    total_figsize = (figsize[0] * 3, figsize[1])
    fig, axes = plt.subplots(1, 3, figsize=total_figsize)

    for i, case in enumerate(cases):
        psc.plot_sliceXY(case=case,
                         mode=mode,
                         target_frequency=target_frequency,
                         absorption=absorption,
                         filter_under=filter_under,
                         filter_over=filter_over,
                         structure=structure,
                         figsize=figsize,
                         cmap=cmap,
                         ax_in=axes[i])
        
        if i > 0:
            axes[i].set_ylabel("")

    # Delete Depth label if it is the same for all plots
    titles = [ax.get_title() for ax in axes]
    
    match = re.search(r'([-\(,\s]*(?:Depth|Z)\s*[:=]\s*[-0-9\.]+\s*m?[\)]*)', titles[0], re.IGNORECASE)
    
    if match:
        depth_str = match.group(1)
        
        if all(depth_str in t for t in titles):
            for ax in axes:
                new_title = ax.get_title().replace(depth_str, '').strip()
                
                if new_title.endswith('-') or new_title.endswith(','):
                    new_title = new_title[:-1].strip()
                    
                ax.set_title(new_title)

    fig.tight_layout()
    return fig, axes

def plot_sliceXZ(cases           : list  = None,      # [-] List of dictionaries containing simulation data
                 mode            : str   = 'OASPL',   # [-] Magnitude to plot: 'OASPL', 'SPL', 'SWL', 'ABS', 'REAL', 'IMAG', 'PHASE'
                 target_frequency: float = 0.88,      # [Hz] Target frequency for single-frequency modes
                 absorption      : bool  = True,      # [-] Whether to apply absorption attenuation
                 filter_under    : float = None,      # [Hz] Lower frequency cutoff (f >= filter_under)
                 filter_over     : float = None,      # [Hz] Upper frequency cutoff (f <= filter_over)
                 structure       : bool  = True,      # [-] Whether to plot structure on slice
                 figsize         : tuple = (7, 7),    # [-] Size of a single panel
                 cmap            : str   = 'inferno'):# [-] Contour colormap
    """
    Plots a 2D spatial XZ slice for three different cases in a 1x3 subplot.
    Expected order in 'cases': [case_monopile, case_floating_shallow, case_floating].
    """

    if cases is None or len(cases) != 3:
        print("plot_sliceXZ(): Exactly 3 cases are required.")
        return None, None

    total_figsize = (figsize[0] * 3, figsize[1])
    
    # ELIMINADO: sharey=True
    fig, axes = plt.subplots(1, 3, figsize=total_figsize)

    for i, case in enumerate(cases):
        psc.plot_sliceXZ(case=case,
                         mode=mode,
                         target_frequency=target_frequency,
                         absorption=absorption,
                         filter_under=filter_under,
                         filter_over=filter_over,
                         structure=structure,
                         figsize=figsize,
                         cmap=cmap,
                         ax_in=axes[i])

        if i > 0: axes[i].set_ylabel("")

    # --- LÓGICA PARA ELIMINAR EL ÁNGULO/AZIMUTH/Y SI ES IDÉNTICO ---
    titles = [ax.get_title() for ax in axes]
    
    # Buscamos un patrón tipo " - Angle: 0 deg", "(Y=0)", "- Azimuth: 90", etc.
    match = re.search(r'([-\(,\s]*(?:Angle|Azimuth|Y)\s*[:=]\s*[-0-9\.]+\s*(?:\[.*?\]|deg|°|m)?[\)]*)', titles[0], re.IGNORECASE)
    
    if match:
        plane_str = match.group(1) # Extraemos exactamente la cadena detectada
        
        # Verificamos si esta cadena exacta está presente en los 3 títulos
        if all(plane_str in t for t in titles):
            for ax in axes:
                # Reemplazamos la cadena del plano por vacío
                new_title = ax.get_title().replace(plane_str, '').strip()
                
                # Limpiamos posibles guiones o comas que hayan quedado al final del título
                if new_title.endswith('-') or new_title.endswith(','):
                    new_title = new_title[:-1].strip()
                    
                ax.set_title(new_title)

    fig.tight_layout()
    return fig, axes
