"""
Module: WindTurbine.py
Description: Abstract Base Class for underwater vibroacoustic modeling of offshore wind turbines.
             This module defines the core data structure and execution flow for acoustic 
             propagation analysis.

Author: Raul Sanz Ramirez (raul.sanz.ramirez@upm.es / raul.sanz.ramirez@gmail.com)
Institution: Universidad Politecnica de Madrid - ETSIAE
Date: 07/2026 
"""

import os
import abc
import numpy as np
from pathlib import Path

class WindTurbine(abc.ABC):
    """
    Abstract Base Class for underwater noise prediction of offshore wind turbines.

    Processing Flow:
    ┌───────────┐      ┌───────────────┐      ┌──────────────┐
    │ OpenFAST  │ ───> │ Coord. Trans. │ ───> │ Subclass API │
    │ I/O Files │      │  (Wind, Pos)  │      │ (Force/Mass) │
    └───────────┘      └───────────────┘      └──────────────┘

    Key Attributes:
    - rootname (str)     : Base name of OpenFAST output files.
    - case_type (str)    : 'Monopile' or 'Floating' (auto-detected).
    - Freqs, F (ndarray) : Frequencies [Hz] and dipole force vectors [N].
    - x_all, x (ndarray) : Structural nodes (Nmembers, Nnodes, 3) vs Wet nodes (Nn, 3) [m].
    - Time, acc (ndarray): Simulation time [s] and acceleration matrix [m/s²].
    - acoustic_data (dic): Internal database for outputs.
    """

    # ========== CONSTRUCTOR ========== #
    def __init__(self,
                 rootname  : str = None,                                # [-] Name without extensions of the OpenFAST output files
                 output_dir: str = "./OP_output/",                      # [-] OpenFAST output directory   
                 save_dir  : str = "./turbine_acoustic_data/",          # [-] Directory to save acoustic results
                 save_name : str = None,                                # [-] Complete name of the acoustic file to save (overrides save_label)
                 WindSpeed : float = None,                              # [m/s] Wind Speed in norm
                 WindDir   : float = 0.0,                               # [deg] Wind direction 0 deg points to +x axis (anticlockwise from +x)
                 Depth     : float = None,                              # [m] Water depth
                 AxisPos   : np.ndarray = np.zeros(2, dtype=float),     # [m] Position of the turbine axis in xy plane
                 BariPos   : np.ndarray = None,                         # [m] Position of the turbine baricenter in xy plane
                 Binary    : bool = True,                               # [-] Whether the OpenFAST output files are in binary format (.outb) or text format (.out)
                 Nmembers  : int = 0,                                   # [-] Number of structural members in the OpenFAST model
                 Nnodes    : int = 0,):                                 # [-] Number of structural nodes in the OpenFAST model
        """
        Initializes physical and simulation configuration parameters.

        Inputs & Config:
        - rootname   [str]        : OpenFAST file name (without extension).
        - output_dir [str]        : Path to OpenFAST output files.
        - save_dir   [str]        : Path to save acoustic results.
        - save_name  [str]        : Custom .npz save name (overrides default).
        - WindSpeed  [float] (m/s): Wind speed magnitude.
        - WindDir    [float] (deg): Wind direction (0° points to +x, CCW).
        - Depth      [float] (m)  : Water column depth.
        - AxisPos    [array] (m)  : 2D coordinates of turbine axis [x, y].
        - BariPos    [array] (m)  : 2D/3D coordinates of turbine baricenter.
        - Binary     [bool]       : Read .outb (True) or .out (False).
        - Method     [str]        : Acoustic propagation model ('images', 'analytic NM').
        - Nmembers   [int]        : SubDyn structural members.
        - Nnodes     [int]        : SubDyn structural nodes per member.
        """

        # Input parameters
        self.rootname   = rootname
        self.WindSpeed  = WindSpeed
        self.WindDir    = WindDir
        self.Depth      = Depth
        self.AxisPos    = AxisPos
        self.BariPos    = BariPos
        self.Nmembers   = Nmembers
        self.Nnodes     = Nnodes

        # Internal attributes
        self.case_type  = None          # [-] States if is a fixed or floating turbine
        self.in_farm    = None          # [-] States if the turbine is in a wind farm or isolated
        self.Freqs      = None          # [Hz] Frequency array for acoustic analysis
        self.F          = None          # [N] Force array for acoustic analysis
        self.x_all      = None          # [m] Coordinates of all the structural nodes (Nmembers, Nnodes, 3)
        self.x          = None          # [m] Coordinates of the acustic sorces linearly (Nn, 3)
        self.Time       = None          # [s] Time array for acoustic analysis
        self.acc        = None          # [m/s^2] Acceleration array for acoustic analysis

        self.acoustic_solver = None     # [-] Acoustic method selection
        self.acoustic_data   = {}       # [-] Dictionary to save results

        # Paths and filenames
        ext              = ".outb" if Binary else ".out"
        self._outfile    = Path(output_dir) / f"{rootname}{ext}"                                        # OpenFAST output file
        self._SUM_SubDyn = Path(output_dir) / f"{rootname}.SD.sum.yaml"                                 # OpenFAST SubDyn sum file
        self._save_path  = Path(save_dir) if save_name is None else Path(save_dir) / f"{save_name}.npz" # Acoustic output file .npz


    # ========== IO FUNCTIONS ========== #
    def read_input(self, 
                   in_farm: bool  = False,      # [-] Wheter WindTurbin is in a WindFarm
                   verbose: bool  = False,      # [-] Flag to print more info
                   skip   : int   = 1,          # [-] Skips time data e.g. Time[::skip]
                   From   : float = 0.,         # [-] Starts reading from here (0. <= From < 1.)
                   Upto   : float = 1.):        # [-] Ends up reading     here (0. < Upto <= 1.)
        """
        Reads structural nodes and acceleration time-series from OpenFAST files.

        Execution Sequence:
        1. Parse SubDyn summary (.yaml) ──> Extract initial node positions (x_all).
        2. Read out/outb files         ──> Extract structural accelerations (acc).
        3. Coord. Transformation       ──> Rotate by WindDir & translate to AxisPos.
        4. Compute Baricenter          ──> Calculate mean of x_all (if not provided).

        Arguments:
        - in_farm (bool)  : True if turbine is inside a wind farm layout.
        - verbose (bool)  : Print ASCII summary of loaded turbine parameters.
        - skip    (int)   : Temporal downsampling step (reads every N-th step).
        - From    (float) : Normalized start time of the signal window [0.0 - 1.0].
        - Upto    (float) : Normalized end time of the signal window [0.0 - 1.0].

        Returns:
        - self (WindTurbine) : Modifies instance coordinates & data in-place.
        """
        
        from utils.IOUtils import get_SDsum_variables, read_input_SD

        self.in_farm = in_farm                      # Only this flag is used on a farm layout

        if self._save_path.suffix == ".npz":        # This path has already been built
            dir = self._save_path.parent
        else:                                       # This path has not been built yet
            dir = self._save_path
        os.makedirs(dir, exist_ok=True)             # Create the directory if it does not exist
        # Load initial position of SubDyn nodes from .yaml file
        if self.Nmembers > 0 and self.Nnodes > 0:
            self.x_all = get_SDsum_variables(self._SUM_SubDyn, Nmembers=self.Nmembers, Nnodes=self.Nnodes, verbose=verbose)
            self.Time, self.acc, _ = read_input_SD(self._outfile, what="acceleration", Nmembers=self.Nmembers, 
                                         Nnodes=self.Nnodes, skip=skip, From=From, Upto=Upto)
        else:
            raise ValueError(f"WindTurbine.read_input(): Nmembers: {self.Nmembers} and Nnodes: {self.Nnodes} are not valid to read SubDyn nodes from .yaml file")
        
        # Compute baricenter position if not provided
        if self.BariPos is None:
            self.BariPos = np.mean(self.x_all[:, :, 0:2], axis=(0, 1))

        # Align with wind direction and translate to AxisPos
        self._translate_align_with_WindDir()

        # Pretty print with turbine info
        if verbose:
            axis_x, axis_y = self.AxisPos[0], self.AxisPos[1]
            dt = self.Time[1] - self.Time[0]
            ws_str = f"{self.WindSpeed:.1f}"
            
            print(f"\n{'='*70}")
            print(f"TURBINE: {self.rootname}")
            print(f"{'─'*68}")
            print(f"Type: {self.case_type:15s}   | ")
            print(f"Wind Speed: {ws_str:>5s} m/s   | Wind Direction: {self.WindDir:3.1f} deg")
            print(f"Water Depth: {self.Depth:4.1f} m   | Position: ({axis_x:4.1f}, {axis_y:4.1f}) m")
            print(f"Time Series: {self.Time.size:6d} samples @ {dt:.4f} s/sample")
            print(f"{'='*70}\n")

        return self
        
    def _translate_align_with_WindDir(self):
        """
        Aligns and positions turbine geometry according to wind direction and farm location.

        Transformations (In-place):
        ┌────────────────────────────────────────────────────────┐
        │  1. ROTATE (Yaw)  : x_all & acc rotated by WindDir     │
        │  2. TRANSLATE     : x_all & BariPos shifted by AxisPos │
        └────────────────────────────────────────────────────────┘

        Returns:
        - self (WindTurbine) : Chaining-friendly reference.
        """

        # --- Build 2-D and 3-D rotation matrices ---
        yaw = np.deg2rad(self.WindDir) 
        c, s = np.cos(yaw), np.sin(yaw)

        R2 = np.array([[c, -s],
                    [s,  c]])   # 2-D rotation (XY plane)

        R3 = np.eye(3)
        R3[:2, :2] = R2            # 3-D rotation (leaves Z untouched)

        axis_xy = np.asarray(self.AxisPos)

        # --- Step 1: rotate nodes around the origin ---
        # x_all starts centred at the origin, so no shift needed before rotation
        self.x_all[..., :2] = self.x_all[..., :2] @ R2.T
        self.acc = self.acc @ R3.T

        # --- Step 2: translate to the turbine axis position ---
        self.x_all[..., :2] += axis_xy

        # --- Apply the same transform to the barycentre ---
        bari = np.asarray(self.BariPos)
        bari[:2] = bari[:2] @ R2.T   # rotate around origin
        bari[:2] += axis_xy           # then translate
        self.BariPos = bari

        return self

    
    # ========== COMPUTE SOURCE TERM ========== #
    @abc.abstractmethod
    def compute_force(self,
                      filter_freqs: bool = False,   # [-] Wheter to skip non inputed frequencis in OpenFAST
                      verbose     : bool = True,    # [-] Flag to print more info
                      skipf       : int = 1):       # [-] Skips frequency data e.g. Freqs[::skipf]
        """
        Compute frequency-domain dipole forces from acceleration data.

        Must be implemented by subclasses. This method updates `self.Freqs`, `self.F`,
        and `self.x` (wet nodes) in place.

        Returns
        -------
        self
            For method chaining.
        """
        pass    # This method is implemented in the subclass of TurbineTypes.py

    @abc.abstractmethod
    def compute_mass(self):
        """
        [Abstract] Calculates structural and hydrodynamic added mass per node.

        Must be implemented by subclasses to define:
        - Structural dry mass distribution.
        - Fluid added mass (Morison equation coefficients).
        """
        pass    # This method is implemented in the subclass of TurbineTypes.py

    @abc.abstractmethod
    def filter_frequencies(self):
        """
        [Abstract] Filters physical frequency bands of the system.

        Must be implemented by subclasses to remove unwanted structural modes 
        (e.g., very low-frequency rigid body motions for floating systems).
        """
        pass    # This method is implemented in the subclass of TurbineTypes.py

    @abc.abstractmethod
    def impedance_correction(self,
                             c_wat: float = 1500):  # [m/s] Speed of sound in fluid. Default: water --> 1500
        """
        Computes an impedance correction for F = gamma*m*a
        """
        pass    # This method is implemented in the subclass of TurbineTypes.py

    
    # ========== OUTPUT MANAGEMENT ========== #
    def save_parameters(self, parameters: dict = None):   # [-] Dictionary with relevant simulation parameters 
        """
        Saves the parameter dictionary provided incrementally in self._save_path.
        """

        if parameters is None: raise ValueError("WindTurbine.save_parameters(): parameters dict not provided")

        self.acoustic_data.update(parameters)
        self.save_acoustics()

        return self

    def save_acoustics(self):
        """
        Saves all computed data incrementally. If the .npz file already exists, keeps old data
        and adds or updates new data.
        """

        # Ensures that the directory exists
        if not self._save_path.parent.exists():
            os.makedirs(self._save_path, exist_ok=True)
        
        if self._save_path.suffix != '.npz':
            raise RuntimeError("WindTurbine.save_acosutics(): save_name has to be declared initializing WindTurbine")

        data_to_save = {**self.acoustic_data}
        
        # If the file already exists, load and update
        if self._save_path.exists():
            existing_data = dict(np.load(self._save_path, allow_pickle=True))
            existing_data.update(data_to_save)
            data_to_save = existing_data

        np.savez(self._save_path, **data_to_save)

        return self

    # ========== SOLVER MANAGEMENT ========== #
    def set_acoustic_method(self, 
                            solver = None): # [-] Which acoustic solver to use
        self.acoustic_solver = solver

        # Save turbine parameters
        parameters = {
            "WindSpeed"      : self.WindSpeed,
            "WindDir"        : self.WindDir,
            "Depth"          : self.Depth,
            "Method"         : solver.get_name() if solver else "None",
            "Structure_nodes": self.x_all,
            "Case_type"      : self.case_type,
            "In_farm"        : self.in_farm,
            "AxisPos"        : self.AxisPos,
            "BariPos"        : self.BariPos,
            "Nm"             : self.Nmembers,
            "Nn"             : self.Nnodes,
            "p_ref"          : solver.p_ref
        }

        # Add solver parameters
        if hasattr(solver, '__dict__'):
            parameters["SolverParams"] = solver.__dict__

        self.save_parameters(parameters)

        return self

    def check_acoustic_solver(self):
        if self.acoustic_solver is None:
            raise RuntimeError("An acoustic solver has to be assigned before pressure computations.")


    # ========== COMPUTE PRESSURE FIELDS ========== #
    def run_spectrums(self, 
                      observers  : np.ndarray = None,   # [m] Observers coordinates array to compute pressure at shape(:,3)
                      z_obs      : float      = None,   # [m] General z coordinate for observers 
                      print_every: int        = 1):     # [-] Print info every specific number of observers
        """
        Compute pressure frequency spectra at specific observer coordinates.

        Uses the current `acoustic_solver` to compute complex pressure at each frequency.
        If `observers` is not provided, defaults to three points (x=10,250,500 m)
        at z = -Depth/2.

        Results are stored in `self.acoustic_data` under:
            - 'P_spectrums' : complex pressure (n_freq, n_obs)
            - 'Obs_spectrums': observer coordinates (n_obs, 3)
            - 'Freqs'        : frequency array (n_freq,)

        Data are automatically saved to the `.npz` file specified by `_save_path`.

        Returns
        -------
        self
            For method chaining.
        """

        self.check_acoustic_solver()
        print("\nComputing spectrums at observer points...")

        # Define observers array
        if observers is None:
            observers = np.zeros((3,3))
            if z_obs is None:
                observers[:,2] = -self.Depth/2.0
            else:
                observers[:,2] = z_obs
            observers[:,0] = [10., 250., 500.]

        # Compute pressure
        nf   = len(self.Freqs)
        Nobs = observers.shape[0] 
        p = np.zeros((nf, Nobs), dtype=complex)
        p = self.acoustic_solver.compute_pressure(self, observers, print_every)
        
        # Save data
        self.acoustic_data["P_spectrums"]   = p
        self.acoustic_data["Obs_spectrums"] = observers
        self.acoustic_data["Freqs"]         = self.Freqs

        self.save_acoustics()
        print(f"\n --> Spectrum data saved at {self._save_path}")
        del self.acoustic_data; self.acoustic_data = {}

        return self

    def run_polar(self,
                  r          : float      = 500.,       # [m] Radius of the circle of observers
                  z          : float      = None,       # [m] z-plane where circle is located
                  n_theta    : int        = 72,         # [-] Number of observers algo the circunference
                  center     : np.ndarray = None,       # [m] Coordinates of the center in z-plane shape (2,)
                  print_every: int        = 10):        # [-] Print info every specific number of observers
        """
        Compute pressure on a circular horizontal ring at fixed depth.

        Uses the current `acoustic_solver`. The ring is defined by radius `r`, depth `z`,
        and centre `center`. Pressure is computed at `n_theta` evenly spaced azimuths.

        Stored in `self.acoustic_data`:
            - 'P_polar'         : complex pressure (n_freq, n_theta)
            - 'R_polar'         : radius (scalar)
            - 'Z_polar'         : depth (scalar)
            - 'Theta_deg_polar' : azimuth angles in degrees (n_theta,)
            - 'Obs_polar'       : observer coordinates (n_theta, 3)
            - 'Center_polar'    : centre coordinates (2,)

        Data are saved to `.npz`.

        Returns
        -------
        self
        """

        # Check defaults
        if center is not None and len(center) != 2: raise ValueError("WindTurbine.run_polar(): center shape should be (2,)")
        if center is None: 
            cx, cy = self.BariPos[0], self.BariPos[1]
        else:
            cx, cy = center[0], center[1]
        if z is None: z = - self.Depth/2.0

        # Build observers array
        theta_deg = np.linspace(0., 360., n_theta, endpoint=False)
        theta_rad = np.deg2rad(theta_deg)
        observers  = np.column_stack([
            r * np.cos(theta_rad) + cx,
            r * np.sin(theta_rad) + cy,
            np.full(n_theta, z)])

        # Compute pressure
        Nf = len(self.Freqs)
        print(f"\nComputing polar pressure (r={r} m, center=({cx:.2f},{cy:.2f}) m, z={z} m), observers={n_theta}...")
        p = np.zeros((Nf, n_theta), dtype=complex)
        p = self.acoustic_solver.compute_pressure(self, observers, print_every)

        # Save data
        self.acoustic_data["Freqs"]           = self.Freqs
        self.acoustic_data["P_polar"]         = p
        self.acoustic_data["R_polar"]         = r
        self.acoustic_data["Z_polar"]         = z
        self.acoustic_data["Theta_deg_polar"] = theta_deg
        self.acoustic_data["Obs_polar"]       = observers
        self.acoustic_data["Center_polar"]    = np.asarray([cx, cy])

        self.save_acoustics()
        print(f"\n --> Polar data saved at {self._save_path}")
        del self.acoustic_data; self.acoustic_data = {}

        return self

    def run_cylinder(self,
                     r      : float      = 500,     # [m] Radius of the circle of observers
                     n_theta: int        = 72,      # [-] Number of observers in azimuthal direction
                     nz     : int        = 20,      # [-] Number of observers in z-direction
                     center : np.ndarray = None,    # [m] Cylinder axis x and y coordinates shape(2,)
                     print_every: int    = 100):    # [-] Print info every specific number of observers
        """
        Compute pressure field on the lateral surface of a vertical cylinder.

        Uses the current `acoustic_solver`. The cylinder has radius `r`, vertical axis through
        `center`, and extends from seabed (z=-Depth) to surface (z=0).
        The grid is `n_theta` azimuthal divisions and `nz` vertical divisions.

        Stored in `self.acoustic_data`:
            - 'P_cylinder'         : complex pressure (n_freq, n_theta, nz)
            - 'R_cylinder'         : radius (scalar)
            - 'Z_cylinder'         : vertical coordinates (nz,)
            - 'Theta_deg_cylinder' : azimuth angles (n_theta,)
            - 'Obs_cylinder'       : observer coordinates (n_theta, nz, 3)
            - 'Center_cylinder'    : centre coordinates (2,)

        Data are saved to `.npz`.

        Returns
        -------
        self
        """

        # Check defaults
        if center is not None and len(center) != 2: raise ValueError("WindTurbine.run_polar(): center shape should be (2,)")
        if center is None: 
            cx, cy = self.BariPos[0], self.BariPos[1]
        else:
            cx, cy = center[0], center[1]

        # Build observers array
        theta_deg = np.linspace(0., 360., n_theta, endpoint=False)
        theta_rad = np.deg2rad(theta_deg)
        z         = np.linspace(-self.Depth, 0., nz)

        THETA, Z = np.meshgrid(theta_rad, z, indexing='ij')
        observers = np.stack([
            r * np.cos(THETA.ravel()) + cx,
            r * np.sin(THETA.ravel()) + cy,
            Z.ravel(),], axis=1)
        
        # Compute pressure
        Nf   = len(self.Freqs)
        Nobs = len(observers)
        print(f"\nComputing cylindrical pressure (r={r} m, center=({cx:.2f},{cy:.2f}) m, grid={n_theta}x{nz}={Nobs} obs)...")
        p    = np.zeros((Nf, Nobs), dtype=complex)
        p    = self.acoustic_solver.compute_pressure(self, observers, print_every)
        
        
        # Reshape to (,n_theta, Nobs, )
        observers = observers.reshape((n_theta, nz, 3))
        p = p.reshape((Nf, n_theta, nz))

        # Diferential of area
        dtheta = 2*np.pi / n_theta
        dz     = (z.max() - z.min())/(nz-1) if nz > 1 else 1.0
        dA     = r * dz * dtheta   

        # Save data
        self.acoustic_data["Freqs"]              = self.Freqs
        self.acoustic_data["P_cylinder"]         = p
        self.acoustic_data["R_cylinder"]         = r
        self.acoustic_data["Z_cylinder"]         = z
        self.acoustic_data["Theta_deg_cylinder"] = theta_deg
        self.acoustic_data["Obs_cylinder"]       = observers
        self.acoustic_data["Center_cylinder"]    = np.asarray([cx, cy])
        self.acoustic_data["dA_cylinder"]        = dA

        self.save_acoustics()
        print(f"\nCylinder data saved at {self._save_path}")
        del self.acoustic_data; self.acoustic_data = {}

        return self

    def run_decay(self,
                  distance   : np.ndarray = [10., 500.],    # [m] Minimum an maximum distance from turbine in x-direction
                  n_points   : int        = 200,            # [-] Number of points in the line
                  z          : float      = None,           # [m] Depth at which line is located
                  logspace   : bool       = True,           # [-] Wheter spacing is logarithmic or linear
                  print_every: int        = 20):            # [-] Print info every specific number of observers
        """
        Compute pressure decay along the wind direction.

        Uses the current `acoustic_solver`. Observers are placed along a line starting at
        `distance[0]` and ending at `distance[1]` m from the turbine axis, in the direction
        of the wind (`WindDir`). The line lies at depth `z` (`-Depth/2` if not given).

        Stored in `self.acoustic_data`:
            - 'P_decay'         : complex pressure (n_freq, n_points)
            - 'Distances_decay' : actual distances from axis (n_points,)
            - 'Z_decay'         : depth (scalar)
            - 'Obs_decay'       : observer coordinates (n_points, 3)
            - 'Logspace_decay'  : bool indicating spacing type

        Data are saved to `.npz`.

        Returns
        -------
        self
        """

        # Check defaults
        if z is None: z = -self.Depth/2.0
        
        # Wind direction
        WindDir_rad = np.deg2rad(self.WindDir)
        wind_unit   = np.array([np.cos(WindDir_rad), np.sin(WindDir_rad)])

        # Distances along the line
        if logspace:
            distances = np.logspace(np.log10(distance[0]), np.log10(distance[1]), n_points)
        else:
            distances = np.linspace(distance[0], distance[1], n_points)

        # Create observers array
        observers = np.zeros((n_points, 3))
        for i, d in enumerate(distances):
            xy = self.AxisPos + d * wind_unit
            observers[i, :] = [xy[0], xy[1], z]

        if np.isnan(observers).any(): raise ValueError("WindTurbine.run_decay(): distance must be grater than 0 to avoid singularities.")

        # Compute pressure
        Nf = len(self.Freqs)
        print(f"\nComputing distance decay (points={n_points}, direction={self.WindDir} deg, z={z} m)...")
        p = np.zeros((Nf, n_points), dtype=complex)
        p = self.acoustic_solver.compute_pressure(self, observers, print_every)

        # Save data
        self.acoustic_data["Freqs"]           = self.Freqs
        self.acoustic_data["P_decay"]         = p
        self.acoustic_data["Z_decay"]         = z
        self.acoustic_data["Distance_decay"]  = distance
        self.acoustic_data["Distances_decay"] = distances
        self.acoustic_data["Obs_decay"]       = observers
        self.acoustic_data["Logspace_decay"]  = logspace

        self.save_acoustics()
        print(f"\nDecay data saved at {self._save_path}")
        del self.acoustic_data; self.acoustic_data = {}

        return self

    def run_line(self,
                 p1         : np.ndarray = None,        # [m] First point coordinate
                 p2         : np.ndarray = None,        # [m] Second point coordinate
                 n_points   : int        = None,        # [-] Number of points along the line
                 logspace   : bool       = False,       # [-] Wheter distribution is logarithmic or linear
                 print_every: int        = None):       # [-] Print info every specific number of observers
        """
        Compute pressure along a straight line between two user-defined points.

        Uses the current `acoustic_solver`. The line is divided into `n_points` equally
        spaced (or logarithmically spaced if `logspace=True`) points between `p1` and `p2`.

        Stored in `self.acoustic_data`:
            - 'P_line'           : complex pressure (n_freq, n_points)
            - 'Distances_line'   : distances from p1 along the line (n_points,)
            - 'P1_line', 'P2_line': endpoints (3,)
            - 'Obs_line'         : observer coordinates (n_points, 3)
            - 'Logspace_line'    : bool

        Data are saved to `.npz`.

        Returns
        -------
        self
        """

        # Check defaults
        if p1 is None: p1 = np.asarray([100.0, 0.0,         0.0])
        if p2 is None: p2 = np.asarray([100.0, 0.0, -self.Depth])
        distance = np.linalg.norm(p2-p1)
        if distance == 0.: raise ValueError("Windturbine.run_line(): points p1 and p2 are coincident.")

        if n_points is None:
            if distance > 200.0:
                n_points = np.floor(distance/5.0)
            else:
                n_points = np.floor(distance)
        print_every = np.floor(n_points/10)
        n_points = int(n_points)

        # Create observers array
        dir_vec = (p2 - p1)/distance
        if logspace:
            distances = np.logspace(np.log10(distance*1e-6), np.log10(distance), n_points)
        else:
            distances = np.linspace(0.0, distance, n_points)
        
        observers = np.zeros((n_points, 3))
        for i, d in enumerate(distances):
            pt = p1 + (d/distance)*dir_vec
            observers[i,:] = pt

        if np.isnan(observers).any(): raise ValueError("WindTurbine.run_line(): Nan within observers array are found")

        # Compute pressure
        Nf = len(self.Freqs)
        print(f"\nComputing line (points={n_points}), p1={p1} m, p2={p2} m)...")
        p = np.zeros((Nf, n_points), dtype=complex)
        p = self.acoustic_solver.compute_pressure(self, observers, print_every)

        # Save data
        self.acoustic_data["Freqs"]          = self.Freqs
        self.acoustic_data["P_line"]         = p
        self.acoustic_data["P1_line"]        = p1
        self.acoustic_data["P2_line"]        = p2
        self.acoustic_data["Obs_line"]       = observers
        self.acoustic_data["Distance_line"]  = distance
        self.acoustic_data["Distances_line"] = distances
        self.acoustic_data["Logspace_line"]  = logspace

        self.save_acoustics()
        print(f"\nLine data saved at {self._save_path}")
        del self.acoustic_data; self.acoustic_data = {}

        return self

    def run_sliceXY(self,
                    z          : float      = None,             # [m] z-plane
                    nx         : int        = 26,               # [-] x-direction discretization
                    ny         : int        = 26,               # [-] y-direction discretization
                    xlim       : np.ndarray = [-500., 500.],    # [m] x-direction range
                    ylim       : np.ndarray = [-500., 500.],    # [m] y-direction range
                    center     : np.ndarray = None,             # [m] Coordinates of the center of the plane
                    print_every: int        = 100):             # [-] Print info every specific number of observers
        """
        Compute pressure field on a horizontal plane at constant depth `z`.

        Uses the current `acoustic_solver`. The plane is defined by `z`, `xlim`, `ylim`,
        and discretised with `nx` by `ny` points. Centre is automatically adjusted.

        Stored in `self.acoustic_data`:
            - 'P_slicexy'   : complex pressure (n_freq, nx, ny)
            - 'X_slicexy'   : x coordinates (nx,)
            - 'Y_slicexy'   : y coordinates (ny,)
            - 'Z_slicexy'   : depth (scalar)
            - 'Obs_slicexy' : observer coordinates (nx, ny, 3)
            - 'Center'      : centre (2,)

        Data are saved to `.npz`.

        Returns
        -------
        self
        """

        # Check defaults
        if z is None: z = -self.Depth/2.0
        if z > 0.0: 
            print("WindTurbine.run_silceXY(): z must be <=  0. Switching to -z")
            z *= -1.
            
        mid_x = 0.5 * (xlim[0] + xlim[1])
        mid_y = 0.5 * (ylim[0] + ylim[1])
        if center is not None:
            center = np.asarray(center, dtype=float).flatten()
            if len(center) != 2:
                raise ValueError("WindTurbine.run_silceXY(): center must contain exactly two elements (x, y).")
            # Check that the provided centre matches the midpoint of the ranges
            if not np.allclose(center, [mid_x, mid_y]):
                print("WindTurbine.run_sliceXY(): provided centre does not match the midpoint of xlim/ylim. "
                    "Switching to the proper centre.")
                center = np.array([mid_x, mid_y])
            else:
                center = np.array([center[0], center[1]])
        else:
            center = np.array([mid_x, mid_y])

        cx, cy = center[0], center[1]

        # Create observers array
        xs = np.linspace(xlim[0], xlim[1], nx)
        ys = np.linspace(ylim[0], ylim[1], ny)

        XX , YY = np.meshgrid(xs, ys)
        observers = np.column_stack((XX.ravel(), YY.ravel(), np.full(nx*ny, z)))
        
        # Compute pressure
        Nf = len(self.Freqs)
        print(f"\nComputing XY slice (nx={nx}, ny={ny}, z={z} m, center=({cx:.1f}, {cy:.1f}) m)...")
        p = np.zeros((Nf,len(observers)), dtype=complex)
        p = self.acoustic_solver.compute_pressure(self, observers, print_every)
        
        # Reshape
        observers = observers.reshape((nx,ny,3))
        p         = p.reshape((Nf, nx, ny))

        # Save data
        self.acoustic_data["Freqs"]          = self.Freqs
        self.acoustic_data["P_slicexy"]      = p
        self.acoustic_data["Obs_slicexy"]    = observers
        self.acoustic_data["X_slicexy"]      = xs
        self.acoustic_data["Y_slicexy"]      = ys
        self.acoustic_data["Z_slicexy"]      = z
        self.acoustic_data["Center_slicexy"] = np.asanyarray([cx, cy])
        self.acoustic_data["Nx_slicexy"]     = nx
        self.acoustic_data["Ny_slicexy"]     = ny

        self.save_acoustics()
        print(f"\nSlice xy data saved at {self._save_path}")
        del self.acoustic_data; self.acoustic_data = {}

        return self

    def run_sliceXZ(self,
                    y          : float      = None,             # [m] Fixed y-coordinate of the slice
                    nx         : int        = 26,               # []- Number of points along x
                    nz         : int        = 26,               # []- Number of points along z (depth)
                    xlim       : np.ndarray = [-500., 500.],    # [m] x-range
                    zlim       : np.ndarray = None,             # [m] z-range (surface to seabed)
                    print_every: int        = 100):             # [-] Print info every specific number of observers
        """
        Compute pressure field on a vertical plane at constant y-coordinate.

        Uses the current `acoustic_solver`. The plane is defined by `y`, `xlim`, `zlim`,
        and discretised with `nx` by `nz` points.

        Stored in `self.acoustic_data`:
            - 'P_slicexz'   : complex pressure (n_freq, nx, nz)
            - 'X_slicexz'   : x coordinates (nx,)
            - 'Z_slicexz'   : z coordinates (nz,)
            - 'Y_slicexz'   : y constant (scalar)
            - 'Obs_slicexz' : observer coordinates (nx, nz, 3)

        Data are saved to `.npz`.

        Returns
        -------
        self
        """

        # Check defaults
        if y is None: y = self.AxisPos[1]
        if zlim is None:
            zlim = np.array([0.0, -self.Depth])
        else:
            zlim = np.asarray(zlim, dtype=float).flatten()
            if zlim.size != 2:
                raise ValueError("zlim must contain exactly two elements (top, bottom).")

        # Create observers array
        xs = np.linspace(xlim[0], xlim[1], nx)
        zs = np.linspace(zlim[0], zlim[1], nz)
        XX, ZZ = np.meshgrid(xs, zs, indexing='ij')

        observers = np.column_stack((
                                    XX.ravel(),
                                    np.full(nx * nz, y),
                                    ZZ.ravel()))

        # Compute pressure
        Nf = len(self.Freqs)
        print(f"\nComputing XZ slice (nx={nx}, nz={nz}, y={y:.1f} m, z from {zlim[0]:.1f} to {zlim[1]:.1f} m)...")
        p = np.zeros((Nf, len(observers)), dtype=complex)
        p = self.acoustic_solver.compute_pressure(self, observers, print_every)

        # Reshape 
        observers = observers.reshape((nx, nz, 3))
        p         = p.reshape((Nf, nx, nz))

        # Save data
        self.acoustic_data["Freqs"]       = self.Freqs
        self.acoustic_data["P_slicexz"]   = p
        self.acoustic_data["Obs_slicexz"] = observers
        self.acoustic_data["X_slicexz"]   = xs
        self.acoustic_data["Z_slicexz"]   = zs
        self.acoustic_data["Y_slicexz"]   = y
        self.acoustic_data["Nx_slicexz"]  = nx
        self.acoustic_data["Nz_slicexz"]  = nz

        self.save_acoustics()
        print(f"\nSlice XZ data saved at {self._save_path}")
        del self.acoustic_data; self.acoustic_data = {}

        return self    

    def run_sphere(self,
                   r          : float      = 30.,       # [m] Radius of the sphere
                   n_theta    : int        = 72,        # [-] Number of points in azimuthal direction
                   nz         : int        = 20,        # [-] Number of vertical layers
                   center     : np.ndarray = None,      # [m] Coordinates of the center
                   print_every: int        = 100):      # [-] Print info every specific number of observers
        """
        Compute pressure on a spherical surface in free-field (no reflections).

        Uses the current `acoustic_solver` but temporarily sets `N_images=0` to disable
        boundary images. The sphere is centred at `center` and radius `r`; if `r` is too
        small to enclose all structural nodes, it is increased automatically.

        The surface is discretised as `n_theta` azimuthal and `nz` vertical layers.

        Stored in `self.acoustic_data`:
            - 'P_sphere'       : complex pressure (n_freq, n_theta, nz)
            - 'Theta_sphere'   : azimuth angles in degrees (n_theta,)
            - 'Z_sphere'       : vertical coordinates (nz,)
            - 'R_sphere'       : radius (scalar)
            - 'Center_sphere'  : centre (3,)
            - 'dA_sphere'      : approximate surface element area (scalar)

        Data are saved to `.npz`.

        Returns
        -------
        self
        """

        # Check defaults
        if center is None:
            cx, cy = self.BariPos[0], self.BariPos[1]
            cz = float(np.mean(self.x[:,2]))
        else:
            center = np.asarray(center, dtype=float).flatten()
            if len(center) == 2:
                cx, cy = center
                cz = float(np.mean(self.pos_0[:, 2]))
            elif len(center) == 3:
                cx, cy, cz = center
            else:
                raise ValueError("WindTurbine.run_sphere(): center must contain 2 or 3 elements (x,y) or (x,y,z).")
        
        # Ensure sphere encloses all structural nodes
        node_dist = np.linalg.norm(self.x - np.array([cx, cy, cz]), axis=1)
        max_dist = node_dist.max() * 1.1
        if r < max_dist:
            print(
                f"WindTurbine.run_sphere(): requested radius r={r:.2f} m is smaller than the largest "
                f"node-centre distance ({max_dist:.2f} m). Radius has been increased "
                f"to {max_dist:.2f} m to enclose all nodes."
            )
            r = max_dist * 1.05

        # Create observers array
        z_edges = np.linspace(cz - r, cz + r, nz + 1)
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])          # nz

        theta_edges = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])   # n_theta

        observers_list = []
        for zc in z_centers:
            r_xy = np.sqrt(max(0.0, r ** 2 - (zc - cz) ** 2))
            x = cx + r_xy * np.cos(theta_centers)
            y = cy + r_xy * np.sin(theta_centers)
            z = np.full_like(theta_centers, zc)
            observers_list.append(np.column_stack([x, y, z]))
        observers = np.concatenate(observers_list, axis=0)      # (n_theta * nz, 3)

        # Comput pressure
        N_obs = observers.shape[0]
        Nf = len(self.Freqs)
        print(f"\nComputing spherical pressure field …")
        print(f"  Centre: ({cx:.2f}, {cy:.2f}, {cz:.2f}) m")
        print(f"  Radius: {r:.2f} m  |  grid: {n_theta} azim x {nz} z  ({N_obs} observers)")
        
        original_n_images = self.acoustic_solver.N_images       # Convert to 0 images for radiation
        self.acoustic_solver.N_images = 0
        p = np.zeros((Nf, N_obs), dtype=complex)
        p = self.acoustic_solver.compute_pressure(self, observers, print_every)
        self.acoustic_solver.N_images = original_n_images

        # Reshape
        observers = observers.reshape((n_theta, nz,3))
        p         = p.reshape((Nf, n_theta, nz))

        # Save data
        self.acoustic_data["Freqs"] = self.Freqs
        self.acoustic_data["P_sphere"]       = p
        self.acoustic_data["Obs_sphere"]     = observers
        self.acoustic_data["Theta_sphere"]   = np.rad2deg(theta_centers) 
        self.acoustic_data["Z_sphere"]       = z_centers           # vertical coordinates
        self.acoustic_data["R_sphere"]       = r                   # scalar radius
        self.acoustic_data["Center_sphere"]  = np.array([cx, cy, cz])
        self.acoustic_data["N_theta"]        = n_theta
        self.acoustic_data["Nz_sphere"]      = nz
        self.acoustic_data["dA_sphere"]      = (4.0 * np.pi * r ** 2) / (n_theta * nz)

        self.save_acoustics()
        print(f"Sphere data saved at {self._save_path}")
        del self.acoustic_data; self.acoustic_data = {}

        return self

    def run_all(self,
                # ---- run_spectrums ----
                spectrums_observers   : np.ndarray = None,
                spectrums_z_obs       : float      = None,
                spectrums_print_every : int        = 1,
                # ---- run_polar ----
                polar_r               : float      = 500.,
                polar_z               : float      = None,
                polar_n_theta         : int        = 72,
                polar_center          : np.ndarray = None,
                polar_print_every     : int        = 10,
                # ---- run_cylinder ----
                cylinder_r            : float      = 500,
                cylinder_n_theta      : int        = 72,
                cylinder_nz           : int        = 20,
                cylinder_center       : np.ndarray = None,
                cylinder_print_every  : int        = 100,
                # ---- run_decay ----
                decay_distance        : np.ndarray = [10., 500.],
                decay_n_points        : int        = 200,
                decay_z               : float      = None,
                decay_logspace        : bool       = True,
                decay_print_every     : int        = 20,
                # ---- run_line ----
                line_p1               : np.ndarray = None,
                line_p2               : np.ndarray = None,
                line_n_points         : int        = None,
                line_logspace         : bool       = False,
                line_print_every      : int        = None,
                # ---- run_sliceXY ----
                sliceXY_z             : float      = None,
                sliceXY_nx            : int        = 26,
                sliceXY_ny            : int        = 26,
                sliceXY_xlim          : np.ndarray = [-500., 500.],
                sliceXY_ylim          : np.ndarray = [-500., 500.],
                sliceXY_center        : np.ndarray = None,
                sliceXY_print_every   : int        = 100,
                # ---- run_sliceXZ ----
                sliceXZ_y             : float      = None,
                sliceXZ_nx            : int        = 26,
                sliceXZ_nz            : int        = 26,
                sliceXZ_xlim          : np.ndarray = [-500., 500.],
                sliceXZ_zlim          : np.ndarray = None,
                sliceXZ_print_every   : int        = 100,
                # ---- run_sphere ----
                sphere_r              : float      = 30.,
                sphere_n_theta        : int        = 72,
                sphere_nz             : int        = 20,
                sphere_center         : np.ndarray = None,
                sphere_print_every    : int        = 100):
        """
        Run all standard acoustic post-processing steps in sequence.

        This method calls `run_spectrums`, `run_polar`, `run_cylinder`, `run_decay`,
        `run_line`, `run_sliceXY`, `run_sliceXZ`, and `run_sphere` with the provided
        parameters (each prefixed by the method name). If a parameter is omitted, the
        default of the corresponding method is used.

        All results are stored in `self.acoustic_data` and saved to the `.npz` file.

        Returns
        -------
        self
        """

        self.check_acoustic_solver()

        total_steps = 8
        print(f"\n{'='*50}")
        print(f" RUNNING ALL ACOUSTIC POST-PROCESSING")
        print(f" Total steps: {total_steps}")
        print(f"{'='*50}")

        # ----- 1. Spectrums -----
        print(f"\n[Step 1/{total_steps}] Spectrums at observer points …")
        self.run_spectrums(
            observers   = spectrums_observers,
            z_obs       = spectrums_z_obs,
            print_every = spectrums_print_every
        )

        # ----- 2. Polar -----
        print(f"\n[Step 2/{total_steps}] Polar contour …")
        self.run_polar(
            r           = polar_r,
            z           = polar_z,
            n_theta     = polar_n_theta,
            center      = polar_center,
            print_every = polar_print_every
        )

        # ----- 3. Cylinder -----
        print(f"\n[Step 3/{total_steps}] Cylinder surface …")
        self.run_cylinder(
            r           = cylinder_r,
            n_theta     = cylinder_n_theta,
            nz          = cylinder_nz,
            center      = cylinder_center,
            print_every = cylinder_print_every
        )

        # ----- 4. Decay -----
        print(f"\n[Step 4/{total_steps}] Distance decay along wind …")
        self.run_decay(
            distance    = decay_distance,
            n_points    = decay_n_points,
            z           = decay_z,
            logspace    = decay_logspace,
            print_every = decay_print_every
        )

        # ----- 5. Line -----
        print(f"\n[Step 5/{total_steps}] Line between two points …")
        self.run_line(
            p1          = line_p1,
            p2          = line_p2,
            n_points    = line_n_points,
            logspace    = line_logspace,
            print_every = line_print_every
        )

        # ----- 6. Slice XY -----
        print(f"\n[Step 6/{total_steps}] Horizontal slice XY …")
        self.run_sliceXY(
            z           = sliceXY_z,
            nx          = sliceXY_nx,
            ny          = sliceXY_ny,
            xlim        = sliceXY_xlim,
            ylim        = sliceXY_ylim,
            center      = sliceXY_center,
            print_every = sliceXY_print_every
        )

        # ----- 7. Slice XZ -----
        print(f"\n[Step 7/{total_steps}] Vertical slice XZ …")
        self.run_sliceXZ(
            y           = sliceXZ_y,
            nx          = sliceXZ_nx,
            nz          = sliceXZ_nz,
            xlim        = sliceXZ_xlim,
            zlim        = sliceXZ_zlim,
            print_every = sliceXZ_print_every
        )

        # ----- 8. Sphere -----
        print(f"\n[Step 8/{total_steps}] Spherical radiation (free-field) …")
        self.run_sphere(
            r           = sphere_r,
            n_theta     = sphere_n_theta,
            nz          = sphere_nz,
            center      = sphere_center,
            print_every = sphere_print_every
        )

        print(f"\n{'='*50}")
        print(f" ALL ACOUSTIC COMPUTATIONS FINISHED")
        print(f" Results saved in: {self._save_path}")
        print(f"{'='*50}")

        return self

