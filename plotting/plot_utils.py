import argparse
import numpy as np
from pathlib import Path
import utils.AcousticUtils as au
import plotting.plot_style as ps
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------- TURBINE UTILITIES ---------- #
def parse_turbine_args():
    """
    Parses arguments from the terminal.
    """

    parser = argparse.ArgumentParser(description="Parse arguments for plotting single wind turbines.")
    parser.add_argument(
        "--data_dir", type = str, default = "./turbine_acoustic_data",
        help = "Directory where the .npz files are stored (default: ./turbine_acoustic_data)"
    )
    parser.add_argument(
        "--method", type = str, default = "images",
        help = "Which acoustic method has been selected (default: images)"
    )
    parser.add_argument(
        "--type", type = str, choices = ["monopile", "floating", "floating_shallow", "comparison"],
        default = "monopile",
        help = "Which case to plot (default: monopile)"
    )
    parser.add_argument(
        "--label", type = str, default = None,
        help = "Optional name tag appended to filenames, e.g. --name ws11.4 (default: None)"
    )
    parser.add_argument(
        "--save", action= "store_true",
        help = "Saves generated plots."
    )
    parser.add_argument(
        "--output_dir", type = str, default = "./figures",
        help = "Path where plots will be saved (default: ./figures)"
    )
    parser.add_argument(    # Only usefull for images method
        "--images", type = int, default = 30,
        help = "Number of image levels used with ImageMethod (default: 30)"
    )

    return parser.parse_args()

def build_turbine_path(args: argparse.Namespace = None):
    
    if args.type == "monopile":
        tag = "mn"
    elif "floating" in args.type:
        tag = "fl"
    else:
        raise ValueError(f"plot_utils.load_case(): non valid args.type = {args.type}")
    if "shallow" in args.type:
        tag_extra = "_shallow"
    else:
        tag_extra = ""

    label_suffix = ""
    if args.label:
        label_suffix = "_" + args.label
        
    if args.method == "images":
        npz_name = "plot_" + f"{tag}_" + f"SD{args.images}" + tag_extra + label_suffix + ".npz"
        npz_name = Path(npz_name)
    else:
        raise ValueError("plot_utils.load_case(): no other method than 'images' is implemented yet")
    
    npz_path = args.data_dir / npz_name
    
    return npz_path

def load_turbine(args   : argparse.Namespace = None,    # [-] Parsed command-line arguments
                 verbose: bool = False):                # [-] if True, prints a list of keys available in the file
    """
    Loads all available data from .npz file coming from args.

    Parameters:
    - args    : parsed command-line arguments
    - verbose : if True, prints an elegant list of keys available in the file
    """
    
    npz_path = build_turbine_path(args)
    print(f"Loading: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)
    if verbose:
        keys = sorted(d.keys())
        print("Available data keys:")
        print("--------------------")
        for idx, key in enumerate(keys, start=1):
            print(f"{idx:2d}. {key}")
        print("--------------------")


    # ---------- Create case dictionary ---------- #
    case = {}
    case["Path"] = npz_path
    case["Case_type"] = d["Case_type"]

    if "shallow" in args.type or "shallow" in str(npz_path):
        case["Case_type"] = "Floating (shallow)"

    # Solver parameters: depends on acoustic method
    if args.method == "images":
        solver_params = d["SolverParams"]
        if isinstance(solver_params, np.ndarray) and solver_params.dtype == object and solver_params.shape == ():
            solver_params = solver_params.item()

        if not isinstance(solver_params, dict):
            raise TypeError("plot_utils.load_case(): SolverParams must be a dict after loading.")

        case["N_images"] = solver_params["N_images"]
        case["c_wat"]    = solver_params["c_wat"]
        case["rho_wat"]  = solver_params["rho_wat"]
    else:
        raise ValueError("plot_utils.load_case(): no other method than 'images' is implemented yet")

    # Common variables
    case["Structure_nodes"] = d["Structure_nodes"]
    case["Depth"]     = d["Depth"]
    case["AxisPos"]   = d["AxisPos"]
    case["BariPos"]   = d["BariPos"]
    case["WindDir"]   = d["WindDir"]
    case["WindSpeed"] = d["WindSpeed"]
    case["p_ref"]     = d["p_ref"]
    case["In_farm"]   = d["In_farm"]
    case["Nm"]        = d["Nm"]
    case["Nn"]        = d["Nn"]
    case["Freqs"]     = d["Freqs"]

    # 1. Spectrum variables
    case["has_spectrums"] = "P_spectrums" in d
    if case["has_spectrums"]:
        case["P_spectrums"]   = d["P_spectrums"]
        case["Obs_spectrums"] = d["Obs_spectrums"]

    # 2. Polar
    case["has_polar"] = "P_polar" in d
    if case["has_polar"]:
        case["P_polar"]         = d["P_polar"]
        case["R_polar"]         = d["R_polar"]
        case["Z_polar"]         = d["Z_polar"]
        case["Obs_polar"]       = d["Obs_polar"]
        case["Theta_deg_polar"] = d["Theta_deg_polar"]
        case["Center_polar"]    = d["Center_polar"]

    # 3. Cylinder
    case["has_cylinder"] = "P_cylinder" in d
    if case["has_cylinder"]:
        case["P_cylinder"]         = d["P_cylinder"]
        case["R_cylinder"]         = d["R_cylinder"]
        case["Z_cylinder"]         = d["Z_cylinder"]
        case["Obs_cylinder"]       = d["Obs_cylinder"]
        case["Theta_deg_cylinder"] = d["Theta_deg_cylinder"]
        case["Center_cylinder"]    = d["Center_cylinder"]

    # 4. Decay
    case["has_decay"] = "P_decay" in d
    if case["has_decay"]:
        case["P_decay"]         = d["P_decay"]
        case["Z_decay"]         = d["Z_decay"]
        case["Distance_decay"]  = d["Distance_decay"]
        case["Distances_decay"] = d["Distances_decay"]
        case["Obs_decay"]       = d["Obs_decay"]
        case["Logspace_decay"]  = d["Logspace_decay"]

    # 5. Line
    case["has_line"] = "P_line" in d
    if case["has_line"]:
        case["P_line"]         = d["P_line"]
        case["P1_line"]        = d["P1_line"]
        case["P2_line"]        = d["P2_line"]
        case["Distance_line"]  = d["Distance_line"]
        case["Distances_line"] = d["Distances_line"]
        case["Obs_line"]       = d["Obs_line"]
        case["Logspace_line"]  = d["Logspace_line"]

    # 6. Slice XY
    case["has_slicexy"] = "P_slicexy" in d
    if case["has_slicexy"]:
        case["P_slicexy"]    = d["P_slicexy"]
        case["Obs_slicexy"]  = d["Obs_slicexy"]
        case["X_slicexy"]    = d["X_slicexy"]
        case["Y_slicexy"]    = d["Y_slicexy"]
        case["Z_slicexy"]    = d["Z_slicexy"]
        case["Center_slicexy"] = d["Center_slicexy"]
        case["Nx_slicexy"]   = d["Nx_slicexy"]
        case["Ny_slicexy"]   = d["Ny_slicexy"]

    # 7. Slice XZ
    case["has_slicexz"] = "P_slicexz" in d
    if case["has_slicexz"]:
        case["P_slicexz"]    = d["P_slicexz"]
        case["Obs_slicexz"]  = d["Obs_slicexz"]
        case["X_slicexz"]    = d["X_slicexz"]
        case["Z_slicexz"]    = d["Z_slicexz"]
        case["Y_slicexz"]    = d["Y_slicexz"]
        case["Nx_slicexz"]   = d["Nx_slicexz"]
        case["Nz_slicexz"]   = d["Nz_slicexz"]

    # 8. Sphere
    case["has_sphere"] = "P_sphere" in d
    if case["has_sphere"]:
        case["P_sphere"]       = d["P_sphere"]
        case["Obs_sphere"]     = d["Obs_sphere"]
        case["Theta_sphere"]   = d["Theta_sphere"]
        case["Z_sphere"]       = d["Z_sphere"]
        case["R_sphere"]       = d["R_sphere"]
        case["Center_sphere"]  = d["Center_sphere"]
        case["N_theta"]        = d["N_theta"]
        case["Nz_sphere"]      = d["Nz_sphere"]
        case["dA_sphere"]      = d["dA_sphere"]

    return case

def load_turbines(args: argparse.Namespace = None,      # [-] Parsed command-line arguments
               verbose: bool = False):                  # [-] if True, prints a list of keys available in the file

    """
    Loads one or all three cases depending on mode or comparison flag.
    """

    if not args.type == "comparison":
        return load_turbine(args, verbose=verbose)
    else:
        args.type = "monopile"
        case_monopile = load_turbine(args, verbose=verbose)
        args.type = "floating"
        case_floating = load_turbine(args, verbose=verbose)
        args.type = "floating_shallow"
        case_floating_shallow = load_turbine(args, verbose=verbose)

        return [case_monopile, case_floating_shallow, case_floating]

        
# ---------- FARM UTILITIES ---------- #
def parse_farm_args():
    """
    Parses arguments from the terminal.
    """
    parser = argparse.ArgumentParser(description="Parse arguments for plotting wind farms.")
    parser.add_argument(
        "--data_dir", type = str, default = "./farm_acoustic_data/",
        help = "Directory where the .npz files are stored (default: ./farm_acoustic_data)"
    )
    parser.add_argument(
        "--name", type = str, default = "Farm_2_DTU10MN.npz"
    )
    parser.add_argument(
        "--save", action= "store_true",
        help = "Saves generated plots."
    )
    parser.add_argument(
        "--output_dir", type = str, default = "./figures",
        help = "Path where plots will be saved (default: ./figures)"
    )

    return parser.parse_args()

def build_farm_path(args: argparse.Namespace = None):

    if args.name:
        npz_path = Path(args.data_dir+args.name)
        if npz_path.exists():
            return npz_path
    else:
        raise RuntimeError("build_farm_path(): path does not exist")

def load_farm(args   : argparse.Namespace = None,   # [-] Parsed command-line arguments
              verbose: bool = False):               # [-] if True, prints a list of keys available in the file
    """
    Loads files from farm simulation
    """

    npz_path = build_farm_path(args)
    print(f"Loading: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)
    if verbose:
        keys = sorted(d.keys())
        print("Available data keys:")
        print("--------------------")
        for idx, key in enumerate(keys, start=1):
            print(f"{idx:2d}. {key}")
        print("--------------------")

    # ---------- Create case dictionary ---------- #
    case = {}
    case["Path"]         = npz_path

    # Farm parameters
    case["Depth"]              = d["Depth"]
    case["Num_turbines"]       = d["Num_Turbines"]
    case["Method"]             = d["Method"]
    case["p_ref"]              = d["p_ref"]
    case["Turbine_parameters"] = d["Turbines"]
    case["Farm_Name"]          = d.get("Farm_Name", npz_path.stem)
    case["SolverParams"]       = d.get("SolverParams", None)

    # Common variables
    case["Freqs"] = d["Freqs"] if "Freqs" in d else None

    # 1. Spectrum variables
    case["has_spectrums"] = "P_spectrums" in d
    if case["has_spectrums"]:
        case["P_spectrums"]   = d["P_spectrums"]
        case["Obs_spectrums"] = d["Obs_spectrums"]

    # 2. Polar
    case["has_polar"] = "P_polar" in d
    if case["has_polar"]:
        case["P_polar"]         = d["P_polar"]
        case["Obs_polar"]       = d["Obs_polar"]
        case["R_polar"]         = d["R_polar"]
        case["Z_polar"]         = d["Z_polar"]
        case["Theta_deg_polar"] = d["Theta_deg_polar"]
        case["Center_polar"]    = d["Center_polar"]

    # 3. Cylinder
    case["has_cylinder"] = "P_cylinder" in d
    if case["has_cylinder"]:
        case["P_cylinder"]         = d["P_cylinder"]
        case["R_cylinder"]         = d["R_cylinder"]
        case["Z_cylinder"]         = d["Z_cylinder"]
        case["Theta_deg_cylinder"] = d["Theta_deg_cylinder"]
        case["Obs_cylinder"]       = d["Obs_cylinder"]
        case["Center_cylinder"]    = d["Center_cylinder"]
        case["dA_cylinder"]        = d.get("dA_cylinder", None)

    # 4. Line
    case["has_line"] = "P_line" in d
    if case["has_line"]:
        case["P_line"]         = d["P_line"]
        case["P1_line"]        = d["P1_line"]
        case["P2_line"]        = d["P2_line"]
        case["Distances_line"] = d["Distances_line"]
        case["Obs_line"]       = d["Obs_line"]
        case["Logspace_line"]  = d["Logspace_line"]

    # 5. Slice XY
    case["has_slicexy"] = ("P_sliceXY" in d) or ("P_slicexy" in d)
    if case["has_slicexy"]:
        case["P_slicexy"]      = d.get("P_sliceXY", d.get("P_slicexy"))
        case["Obs_slicexy"]    = d.get("Obs_slicexy", None)
        case["X_slicexy"]      = d.get("X_sliceXY", d.get("X_slicexy"))
        case["Y_slicexy"]      = d.get("Y_sliceXY", d.get("Y_slicexy"))
        case["Z_slicexy"]      = d.get("Z_sliceXY", d.get("Z_slicexy"))
        case["Center_slicexy"] = d.get("Center_sliceXY", d.get("Center_slicexy"))

    # 6. Slice XZ
    case["has_slicexz"] = ("P_sliceXZ" in d) or ("P_slicexz" in d)
    if case["has_slicexz"]:
        case["P_slicexz"]      = d.get("P_sliceXZ", d.get("P_slicexz"))
        case["Obs_slicexz"]    = d.get("Obs_slicexz", None)
        case["X_slicexz"]      = d.get("X_sliceXZ", d.get("X_slicexz"))
        case["Z_slicexz"]      = d.get("Z_sliceXZ", d.get("Z_slicexz"))
        case["Y_slicexz"]      = d.get("Y_sliceXZ", d.get("Y_slicexz"))
        case["Center_slicexz"] = d.get("Center_sliceXZ", d.get("Center_slicexz"))

    # 7. Vertical slice
    case["has_slicevertical"] = "P_sliceVertical" in d
    if case["has_slicevertical"]:
        case["P_sliceV"]       = d["P_sliceVertical"]
        case["Coords_sliceV"]  = d["Coords_sliceV"]
        case["U_sliceV"]       = d["U_sliceV"]
        case["Z_sliceV"]       = d["Z_sliceV"]
        case["Azimuth_sliceV"] = d["Azimuth_sliceV"]
        case["Center_sliceV"]  = d["Center_sliceV"]

    # 8. Spheres
    case["has_spheres"] = "P_spheres" in d
    if case["has_spheres"]:
        case["P_spheres"]       = d["P_spheres"]
        case["Obs_spheres"]     = d["Obs_spheres"]
        case["Centers_spheres"] = d["Centers_spheres"]
        case["Radii_spheres"]   = d["Radii_spheres"]
        case["N_theta"]         = d["N_theta"]
        case["Nz_sphere"]       = d["Nz_sphere"]
        case["dA_spheres"]      = d["dA_spheres"]

    return case

    
# ---------- COMMON UTILITIES ---------- #
def draw_direction_arrow(ax,                                # [-] Target axis. 3D vs 2D is auto-detected.
                          x              : float,           # [m] Tip location (in the horizontal plane).
                          y              : float,           # [m] Tip location (in the horizontal plane).
                          direction_deg  : float,           # [deg] Arrow direction
                          length         : float,           # [m] Arrow length. Must be > 0.
                          z              : float = 0.0,     # [m] Tip Z coordinate. Ignored for 2D axes.
                          color          : str   = 'red',   # [-] Arrow color
                          label          : str   = None,    # [.] Arrow label
                          head_frac      : float = 0.25,    # [-] Fraction of 'length' used for the arrowhead ignored in 2D
                          head_width_frac: float = 0.25,    # [-] Arrowhead half-width as a fraction of head length ignored in 2D
                          linewidth      : float = 1.8):    # [-] Arrow line width
    """
    Draws a direction arrow (e.g. wind direction) on a 2D or 3D matplotlib axis.
    The arrow tip is placed at (x, y[, z]); it points backward from there
    along 'direction_deg' (degrees, standard math convention: 0=+x, 90=+y).

    Returns
    -------
    The created artist (FancyArrow for 2D, Poly3DCollection for 3D).
    """

    if length is None or length <= 0:
        raise ValueError(f"draw_direction_arrow(): length must be > 0, got {length!r}")

    theta = np.deg2rad(direction_deg)
    u, v = np.cos(theta), np.sin(theta)

    is_3d = isinstance(ax, Axes3D)

    if not is_3d:
        # 2D case: matplotlib's arrow/annotate already draws a proper filled head.
        tail_x = x - length * u
        tail_y = y - length * v
        arrow = ax.annotate(
            '', xy=(x, y), xytext=(tail_x, tail_y),
            arrowprops=dict(arrowstyle='-|>', color=color, linewidth=linewidth,
                             mutation_scale=15),
        )
        if label is not None:
            # dummy line for legend, since annotate() doesn't register one
            ax.plot([], [], color=color, linewidth=linewidth, label=label)
        return arrow

    # 3D case: draw shaft + solid triangular head manually.
    tip = np.array([x, y, z])
    tail = tip - length * np.array([u, v, 0])

    ax.plot([tail[0], tip[0]], [tail[1], tip[1]], [tail[2], tip[2]],
            color=color, linewidth=linewidth)

    head_len = length * head_frac
    head_width = head_len * head_width_frac
    perp = np.array([-v, u, 0])  # in-plane perpendicular to arrow direction
    base_center = tip - head_len * np.array([u, v, 0])
    p1 = tip
    p2 = base_center + head_width * perp
    p3 = base_center - head_width * perp

    head = Poly3DCollection([[p1, p2, p3]], color=color)
    ax.add_collection3d(head)

    if label is not None:
        ax.plot([], [], color=color, linewidth=linewidth, label=label)

    return head

def get_case_color(case: dict = None,           # [-] Dictionary containing simulation data
                   tags: list = None):      # [-] List with tags
    """
    Gets color based on case type.
    """

    def color_from_tag(tag: str):
        if "Monopile" in tag:
            return ps.AcademicStyle.CASE_COLORS[0]
        if "shallow" in tag:
            return ps.AcademicStyle.CASE_COLORS[1]
        return ps.AcademicStyle.CASE_COLORS[2]

    if case is not None:
        return color_from_tag(case["Case_type"])

    if tags is not None:
        return np.array([color_from_tag(tag) for tag in tags])

    raise ValueError("get_case_color(): either 'case' or 'tags' must be provided")

def get_line_styles():
    return ps.AcademicStyle.LINE_STYLES

def get_band_colors():  
    return ps.AcademicStyle.BAND_COLORS

def convert_to(p,                       # [Hz] Pressure array shape(NFreqs, Nobs)
               mode: str = "SPL"):    # [-] What to convert to
    """
    Converts pressure into desired units.
    """

    mode = mode.upper()

    if mode == "SPL":
        vals = au.pressure_to_SPL(p)
        unit_label = r"SPL [dB re 1 $\mu$Pa]"
    elif mode == "ABS":
        vals = np.abs(p)
        unit_label = r"Magnitude [Pa]"
    elif mode == "REAL":
        vals = np.real(p)
        unit_label = r"Real Part [Pa]"
    elif mode == "IMAG":
        vals = np.imag(p)
        unit_label = r"Imaginary Part [Pa]"
    elif mode == "PHASE":
        vals = np.angle(p, deg=True)
        if np.any(vals < -90) and np.any(vals > 90):
            vals = np.mod(vals, 360)
        unit_label = r"Phase [deg]"
    else:
        print(f"plot_polar(): Unrecognized mode '{mode}', defaulting to 'SPL'")
        vals = au.pressure_to_SPL(p)
        unit_label = r"SPL [db re 1 $\mu$Pa]"

    return vals, unit_label

def get_tubine_styles():
    from utils.EnvironmentalData import Turbine_noise
    Tripile, Gravity_based, Monopile, Jacket, Suction_bucket = Turbine_noise()
    TURBINE_STYLES = [
        (Tripile,        "Tripile (Ge 2025)",        "s", "gold"      ),
        (Gravity_based,  "Gravity based (Ge 2025)",  "o", "limegreen" ),
        (Monopile,       "Monopile (Ge 2025)",       "^", "darkred"   ),
        (Jacket,         "Jacket (Ge 2025)",         "v", "steelblue" ),
        (Suction_bucket, "Suction bucket (Ge 2025)", "D", "silver"    ),
    ]

    return TURBINE_STYLES
