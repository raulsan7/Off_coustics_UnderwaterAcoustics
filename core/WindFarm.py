"""
Module: WindFarm.py
Description: Class for managing multiple offshore wind turbines in a farm layout 
             for underwater vibroacoustic modeling.

Author: Raul Sanz Ramirez (raul.sanz.ramirez@upm.es / raul.sanz.ramirez@gmail.com)
Institution: Universidad Politecnica de Madrid - ETSIAE
Date: 08/2026 
"""

import gc
import os
import numpy as np
from pathlib import Path
from collections import Counter

class WindFarm():
    """
    Manager class for a collection of WindTurbine objects.
    
    This class orchestrates the initialization, input reading, and acoustic 
    computations for multiple turbines, ensuring domain consistency (e.g., Depth)
    and generating aggregated output files.
    """

    # ========== CONSTRUCTOR ========== #
    def __init__(self,
                 turbines : list = None,
                 debug    : bool = False,
                 save_dir : str  = "./farm_acoustic_data/",
                 save_name: str  = None):

        if turbines is None: raise ValueError("WindFarm must be initialized with a list of WindTurbine objects.")

        # Input parameters
        self.turbines = turbines
        self.debug    = debug

        # Validate shared domain parameters
        self.Depth = self.turbines[0].Depth
        for i, t in enumerate(self.turbines):
            if t.Depth != self.Depth:
                raise ValueError(f"Turbine {i} ({t.rootname}) has Depth={t.Depth}m, "
                                 f"but Farm Depth is {self.Depth}m. All turbines must share the same Depth.")

        # Generate output filename
        if save_name is None:
            self.save_name = self._generate_save_name()
        else:
            self.save_name = save_name

        self._save_path = Path(save_dir) / f"{self.save_name}.npz"

        # Placegolders for future integration
        self.acoustic_solver = None
        self.acoustic_data   = {}


    # ========== HELPERS ========== #
    def _generate_save_name(self):
        """
        Generates a descriptive filename based on the types and quantities
        of turbines present in the farm.
        Example: 'Farm_2_DTU10MN_1_SAITEC2FL'
        """

        turbine_types = [t.__class__.__name__ for t in self.turbines]
        counts = Counter(turbine_types)

        type_aliases = {
            "DTU10MWMonopile": "DTU10MN",
            "DTU10MWFloating": "DTU10FL",
            "SAITEC2MWFloating": "SAITEC2FL",
        }

        name_parts = ["Farm"]
        for t_type, count in counts.items():
            if t_type not in type_aliases:
                raise ValueError(f"Missing type_aliases entry for turbine type '{t_type}'. Please add a new alias.")

            short_type = type_aliases[t_type]
            name_parts.append(f"{count}_{short_type}")

        return "_".join(name_parts)

    def _check_freq_consistency(self) -> bool:
        """
        Checks if all turbines share identical frequency vectors (Freqs).
        """
        ref_freqs = getattr(self.turbines[0], "Freqs", None)
        if ref_freqs is None:
            return False

        for turbine in self.turbines[1:]:
            t_freqs = getattr(turbine, "Freqs", None)
            if t_freqs is None or len(t_freqs) != len(ref_freqs):
                return False
            # Numerical comparison with tolerance to avoid float inconsitencies
            if not np.allclose(t_freqs, ref_freqs, rtol=1e-5, atol=1e-8):
                return False

        return True

    def check_acoustic_solver(self):
        if self.acoustic_solver is None:
            raise RuntimeError("An acoustic solver has to be assigned before pressure computations.")

    def check_observers_distances(self,
                                  observers: np.ndarray,  # [m] Observers coordinates array shape(Nobs,3)
                                  min_distance: float = 1.0):  # [m] Minimum allowed distance
        """
        Checks if observers are at least min_distance away from any turbine.
        """
        
        all_nodes = np.vstack([t.x for t in self.turbines])

        for i, obs in enumerate(observers):
            distances = np.linalg.norm(all_nodes - obs, axis=1)
            if np.any(distances < min_distance):
                raise ValueError(f"Observer {i} at {obs} is too close to a turbine node (min distance: {min_distance} m).")

        return self


    # ========== IO FUNCTIONS ========== #
    def read_input(self, 
                   verbose: bool  = False,      # [-] Flag to print more info
                   skip   : int   = 1,          # [-] Skips time data e.g. Time[::skip]
                   From   : float = 0.,         # [-] Starts reading from here (0. <= From < 1.)
                   Upto   : float = 1.):        # [-] Ends up reading here (0. < Upto <= 1.)
        """
        Iterates over all turbines and calls their respective read_input() methods.
        """

        if verbose:
            print(f"\n{'='*80}")
            print(f"INITIALIZING WIND FARM: {self.save_name}")
            print(f"{'─'*78}")
            print(f"Total Turbines: {len(self.turbines)}")
            print(f"Farm Depth:     {self.Depth} m")
            print(f"Save Path:      {self._save_path}")
            print(f"{'='*80}\n")

        for i, turbine in enumerate(self.turbines):
            if verbose:
                print(f"--- Loading Turbine {i+1}/{len(self.turbines)}: {turbine.rootname} ({turbine.__class__.__name__}) ---")
            
            # Call the turbine's read_input, explicitly setting in_farm=True
            turbine.read_input(in_farm=True, 
                               verbose=False, 
                               skip=skip, 
                               From=From, 
                               Upto=Upto)

        if verbose:
            print(f"{'-'*80}")
            print("Input loading completed for all turbines in the farm.")
            print("\nTurbine summary:")
            for turbine in self.turbines:
                print(
                    f"  - Type: {turbine.__class__.__name__} | WindDir: {turbine.WindDir} | WindSpeed: {turbine.WindSpeed} | "
                    f"AxisPos: {turbine.AxisPos}"
                )
            print()
            
        return self

    def save_parameters(self, 
                        farm_params: dict):  # [-] Dictionary with relevant simulation parameters
        """
        Saves the parameter dictionary provided incrementally in self._save_path.
        """

        if farm_params is None: 
            raise ValueError("WindFarm.save_parameters(): parameters dict not provided")

        self.acoustic_data.update(farm_params)
        self.save_acoustics()

        return self

    def save_acoustics(self):
        """
        Saves all computed data incrementally. If the .npz file already exists, keeps old data
        and adds or updates new data.
        """

        # Ensures that the parent directory exists
        if not self._save_path.parent.exists():
            os.makedirs(self._save_path.parent, exist_ok=True)

        if self._save_path.suffix != ".npz":
            raise RuntimeError("WindFarm.save_acoustics(): save_name has to be declared initializing WindFarm")

        data_to_save = {**self.acoustic_data}

        # If the file already exists, load and update
        if self._save_path.exists():
            existing_data = dict(np.load(self._save_path, allow_pickle=True))
            existing_data.update(data_to_save)
            data_to_save = existing_data

        np.savez(self._save_path, **data_to_save)

        return self    


    # ========== COMPUTE_FORCE ========== #
    def compute_force(self,
                      verbose     : bool = False,   # [-] Flag to print more info
                      filter_freqs: bool = False,   # [-] Wheter to skip non inputed frequencis in OpenFAST
                      skipf       : int = 1):       # [-] Skips frequency data e.g. Freqs[::skipf]
        """
        Iterates over all turbines in the farm and computes the acoustic excitation forces 
        for each one by calling their respective compute_force() methods.
        """

        if verbose:
            print(f"\n{'='*80}")
            print(f"COMPUTING FORCES FOR WIND FARM: {self.save_name}")
            print(f"{'='*80}")

        # First attempt: compute forces with requested parameters
        for i, turbine in enumerate(self.turbines):
            if verbose:
                print(f"--- Computing forces for Turbine {i+1}/{len(self.turbines)}: {turbine.rootname} ---")
            
            turbine.compute_force(verbose=False, filter_freqs=filter_freqs, skipf=skipf)

        # Check frequency consistency across all turbines
        freqs_match = self._check_freq_consistency()

        # Fallback: if filtering frequencies created mismatch, recalculate with filter_freqs=False
        if not freqs_match and filter_freqs:
            if verbose:
                print("\n[WARNING] Frequency vectors mismatch across turbines when filter_freqs=True.")
                print("          Retrying force computation with filter_freqs=False for all turbines...")

            for turbine in self.turbines:
                turbine.compute_force(verbose=False, filter_freqs=False, skipf=skipf)

            freqs_match = self._check_freq_consistency()

        # If they STILL don't match, raise ValueError
        if not freqs_match:
            raise ValueError(
                "Frequency vectors (Freqs) do not match across all turbines in the farm. "
                "Ensure all turbines share the same time step (dt), total duration, and skip parameters."
            )

        # Store global frequency array at the farm level
        self.Freqs = self.turbines[0].Freqs
        
        if verbose:
            print(f"{'-'*80}")
            print("Force computation completed for all turbines in the farm.")
            print("\nForce summary per turbine:")
            for turbine in self.turbines:
                n_nodes = getattr(turbine, "x", None)
                n_nodes_wet = n_nodes.shape[0] if n_nodes is not None else "N/A"
                n_freqs = len(turbine.Freqs) if getattr(turbine, "Freqs", None) is not None else "N/A"
                force_memory_mb = turbine.F.nbytes / 1024.0 / 1024.0 if getattr(turbine, "F", None) is not None else "N/A"
                print(f"  - {turbine.rootname} | Nodes (wet): {n_nodes_wet} | Frequencies: {n_freqs} | Force array memory: {force_memory_mb:.2f} MB" if isinstance(force_memory_mb, float) else f"  - {turbine.rootname} | Nodes (wet): {n_nodes_wet} | Frequencies: {n_freqs} | Force array memory: {force_memory_mb}")
            print()

        return self


    # ========== SOLVER MANAGEMENT ========== #
    def set_acoustic_solver(self, 
                            solver = None):     # [-] Which acoustic solver to use
        """
        Assigns an acoustic solver to the farm and collects metadata 
        from both the solver and all individual turbines for post-processing and saving.
        """
        self.acoustic_solver = solver

        # Collect individual turbine parameters as a llist of dicts
        turbine_parameters = []
        for i, t in enumerate(self.turbines):
            turbine_params = {
                "Index": i,
                "Rootname": t.rootname,
                "Type": t.__class__.__name__,
                "WindSpeed": t.WindSpeed,
                "WindDir": t.WindDir,
                "Depth": t.Depth,
                "AxisPos": t.AxisPos,
                "BariPos": t.BariPos,
                "Structure_nodes": t.x_all,
                "Case_type": t.case_type,
                "Nm": t.Nmembers,
                "Nn": t.Nnodes,
            }
            turbine_parameters.append(turbine_params)

        # Collect global farm parameters
        farm_params = {
            "Farm_Name": self.save_name,
            "Depth": self.Depth,
            "Num_Turbines": len(self.turbines),
            "Method": solver.get_name() if solver else "None",
            "p_ref": solver.p_ref if solver else 1e-6,
            "Turbines": turbine_parameters
        }

        # Add global solver parameters if available
        if solver and hasattr(solver, "__dict__"):
            solver_dict = {k: v for k, v in solver.__dict__.items() if k != "farm"}
            farm_params["SolverParams"] = solver_dict

        # Store in instance and save to file/metadata
        self.farm_params = farm_params
        self.save_parameters(farm_params)

        return self


    # ========== COMPUTE PRESSURE FIELDS ========== #
    def run_spectrums(self,
                      observers: np.ndarray = None,     # [m] Observers coordinates array to compute pressure at shape(:,3)
                      z_obs    : float      = None,     # [m] General z coordinate for observers 
                      block_size: int       = 4):       # [-] Print info every specific number of observers
        """
        Computes the acoustic pressure spectrum at the specified receiver coordinates 
        by superposing the complex pressure contributions of all turbines in the farm.
        """

        self.check_acoustic_solver()
        print("\nComputing spectrums at observer points...")

        if z_obs is None: z_obs = -self.Depth / 2.0 
        if observers is None:
            AxisPos_mean = np.mean([t.AxisPos for t in self.turbines], axis=0)
            observers = np.zeros((1, 3))
            observers[0, 0] = AxisPos_mean[0]
            observers[0, 1] = AxisPos_mean[1]
            observers[0, 2] = z_obs

        # Check distances to avoid singularities
        self.check_observers_distances(observers)

        Nobs = observers.shape[0]
        Nfreqs = len(self.Freqs)
        p = np.zeros((Nfreqs, Nobs), dtype=complex)
        p = self.acoustic_solver.compute_pressure(observers, block_size)

        if self.debug: print("Spectrums Pressure Norm: ", np.linalg.norm(p))

        # Save data
        self.acoustic_data["P_spectrums"] = p
        self.acoustic_data["Obs_spectrums"] = observers
        self.acoustic_data["Freqs"] = self.Freqs

        self.save_acoustics()
        print(f"\n --> Spectrum data saved at {self._save_path}")

        self.acoustic_data.clear(); gc.collect()
        
        return self

    def run_polar(self,
                  r         : float      = 500.,    # [m] Radius of the circle of observers
                  z         : float      = None,    # [m] z-plane where circle is located
                  n_theta   : int        = 72,      # [-] Number of observers algo the circunference
                  center    : np.ndarray = None,    # [m] Coordinates of the center in z-plane shape (2,)
                  block_size: int        = 16):     # [-] Print info every specific number of observers

        self.check_acoustic_solver()
        if z is None: z = -self.Depth / 2.0
        if center is None: center = np.mean([t.AxisPos for t in self.turbines], axis=0)

        # Build observers array
        theta = np.linspace(0, 360., n_theta, endpoint=False)
        theta_rad = np.deg2rad(theta)
        observers = np.column_stack([
            r * np.cos(theta_rad) + center[0],
            r * np.sin(theta_rad) + center[1],
            np.full(n_theta, z)
        ])

        # Check distances to avoid singularities
        self.check_observers_distances(observers)

        # Compute pressure
        Nfreqs = len(self.Freqs)
        print(f"\nComputing polar pressure (r={r} m, center=({center[0]:.2f},{center[1]:.2f}) m, z={z} m), observers={n_theta}...")
        p = np.zeros((Nfreqs, n_theta), dtype=complex)
        p = self.acoustic_solver.compute_pressure(observers, block_size)

        if self.debug: print("Polar Pressure Norm: ", np.linalg.norm(p))

        # Save data
        self.acoustic_data["Freqs"]           = self.Freqs
        self.acoustic_data["P_polar"]         = p
        self.acoustic_data["R_polar"]         = r
        self.acoustic_data["Z_polar"]         = z
        self.acoustic_data["Theta_deg_polar"] = theta
        self.acoustic_data["Obs_polar"]       = observers
        self.acoustic_data["Center_polar"]    = center

        self.save_acoustics()
        print(f"\n --> Polar data saved at {self._save_path}")

        self.acoustic_data.clear(); gc.collect()

        return self

    def run_cylinder(self,
                     r         : float      = 500.,    # [m] Radius of the cylinder
                     z_start   : float      = None,    # [m] Lower z-coordinate of the cylinder (default: -Depth)
                     z_end     : float      = 0.0,     # [m] Upper z-coordinate of the cylinder (default: surface)
                     n_theta   : int        = 72,      # [-] Number of observers along the circumference
                     nz        : int        = 20,      # [-] Number of observers along the z-axis
                     center    : np.ndarray = None,    # [m] Coordinates of the center in z-plane shape (2,)
                     block_size: int        = 256):     # [-] Number of observers to process simultaneously

        self.check_acoustic_solver()
        
        if z_start is None: z_start = -self.Depth
        if center is None: center = np.mean([t.AxisPos for t in self.turbines], axis=0)

        # Build observers array (Cylindrical mesh)
        theta = np.linspace(0, 360., n_theta, endpoint=False)
        z = np.linspace(z_start, z_end, nz)
        
        theta_rad = np.deg2rad(theta)
        Theta_mesh, Z_mesh = np.meshgrid(theta_rad, z, indexing='ij')
        
        observers = np.column_stack([
            r * np.cos(Theta_mesh.ravel()) + center[0],
            r * np.sin(Theta_mesh.ravel()) + center[1],
            Z_mesh.ravel()
        ])

        # Check distances to avoid singularities
        self.check_observers_distances(observers, min_distance=1.0)

        # Compute pressure
        Nfreqs = len(self.Freqs)
        Nobs = observers.shape[0]
        print(f"\nComputing cylinder pressure (r={r} m, z=[{z_start:.2f}, {z_end:.2f}] m), total observers={Nobs}...")
        
        p = np.zeros((Nfreqs, Nobs), dtype=complex)
        p = self.acoustic_solver.compute_pressure(observers, block_size)

        if self.debug: print("Cylinder Pressure Norm: ", np.linalg.norm(p))

        # Reshape
        observers = observers.reshape((nz, n_theta, 3))
        p = p.reshape((Nfreqs, nz, n_theta))

        # Diferential of area
        dtheta = 2*np.pi / n_theta
        dz     = (z.max() - z.min())/(nz-1) if nz > 1 else 1.0
        dA     = r * dz * dtheta   

        # Save data
        self.acoustic_data["Freqs"]              = self.Freqs
        self.acoustic_data["P_cylinder"]         = p
        self.acoustic_data["R_cylinder"]         = r
        self.acoustic_data["Z_cylinder"]         = z
        self.acoustic_data["Theta_deg_cylinder"] = theta
        self.acoustic_data["Obs_cylinder"]       = observers
        self.acoustic_data["Center_cylinder"]    = center
        self.acoustic_data["dA_cylinder"]        = dA

        self.save_acoustics()
        print(f"\n --> Cylinder data saved at {self._save_path}")

        self.acoustic_data.clear(); gc.collect()

        return self

    def run_line(self,
                 p1        : np.ndarray = None,       # [m] Start point of the line shape (3,)
                 p2        : np.ndarray = None,       # [m] End point of the line shape (3,)
                 n_points  : int        = 200,        # [-] Number of observers along the line
                 logspace  : bool       = False,      # [-] Type of spacing: 'linspace' or 'logspace'
                 block_size: int        = 64):        # [-] Number of observers to process simultaneously

        self.check_acoustic_solver()
        
        # Calculate mean center for default points
        if p1 is None or p2 is None:
            center = np.mean([t.AxisPos for t in self.turbines], axis=0)
            if p1 is None: p1 = np.array([center[0], center[1], 0.0])           # Surface
            if p2 is None: p2 = np.array([center[0], center[1], -self.Depth])   # Seabed

        # Ensure inputs are numpy arrays
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)

        # Generate parametric distribution 's' from 0 to 1
        if logspace:
            # Logarithmic distribution (higher density near p1)
            # We use base 10 from 10^0 to 10^1 and map it to [0, 1]
            s = (np.logspace(0, 1, n_points) - 1.0) / 9.0
        else:
            # Linear distribution
            s = np.linspace(0.0, 1.0, n_points)

        # Build observers array (Vectorized interpolation)
        observers = p1 + s[:, np.newaxis] * (p2 - p1)

        # Check distances to avoid singularities
        self.check_observers_distances(observers, min_distance=1.0)

        # Compute pressure
        Nfreqs = len(self.Freqs)
        Nobs = observers.shape[0]
        print(f"\nComputing line pressure from {p1} to {p2} (logspace: {logspace}), total observers={Nobs}...")
        
        p = np.zeros((Nfreqs, Nobs), dtype=complex)
        p = self.acoustic_solver.compute_pressure(observers, block_size)

        if self.debug: print("Line Pressure Norm: ", np.linalg.norm(p))

        # Distance along the line (useful for plotting)
        line_length = np.linalg.norm(p2 - p1)
        distances_along_line = s * line_length

        # Save data
        self.acoustic_data["Freqs"]             = self.Freqs
        self.acoustic_data["P_line"]            = p
        self.acoustic_data["Obs_line"]          = observers
        self.acoustic_data["Distances_line"]    = distances_along_line
        self.acoustic_data["P1_line"]           = p1
        self.acoustic_data["P2_line"]           = p2
        self.acoustic_data["Logspace_line"]     = logspace

        self.save_acoustics()
        print(f"\n --> Line data saved at {self._save_path}")

        self.acoustic_data.clear(); gc.collect()

        return self

    def run_sliceXY(self,
                    z_slice   : float      = None,    # [m] Depth of the horizontal plane
                    nx        : int        = None,    # [-] Number of points in x-direction
                    ny        : int        = None,    # [-] Number of points in y-direction
                    xlim      : list       = None,    # [m] Limits [x_min, x_max] relative to center
                    ylim      : list       = None,    # [m] Limits [y_min, y_max] relative to center
                    center    : np.ndarray = None,    # [m] Coordinates of the center in z-plane shape (2,)
                    margin    : float      = 0.1,     # [-] Relative margin for auto limits (0.1 = 10%)
                    block_size: int        = 256):     # [-] Number of observers to process simultaneously

        self.check_acoustic_solver()
        
        if z_slice is None: z_slice = -self.Depth / 2.0
        if center is None: center = np.mean([t.AxisPos for t in self.turbines], axis=0)[:2]

        cx, cy = center[0], center[1]
        
        # Extract all nodes to compute auto-limits
        all_nodes_xy = np.vstack([t.x for t in self.turbines])[:, :2]

        # Calculate bounding box
        if xlim is None and ylim is None:
            dist = np.linalg.norm(all_nodes_xy - np.array([cx, cy]), axis=1)
            half_span = (1.0 + margin) * dist.max()
            x_min, x_max = cx - half_span, cx + half_span
            y_min, y_max = cy - half_span, cy + half_span
        elif xlim is None:
            y_min, y_max = cy + ylim[0], cy + ylim[1]
            x_min, x_max = all_nodes_xy[:, 0].min() - margin, all_nodes_xy[:, 0].max() + margin
        elif ylim is None:
            x_min, x_max = cx + xlim[0], cx + xlim[1]
            y_min, y_max = all_nodes_xy[:, 1].min() - margin, all_nodes_xy[:, 1].max() + margin
        else:
            x_min, x_max = cx + xlim[0], cx + xlim[1]
            y_min, y_max = cy + ylim[0], cy + ylim[1]

        range_x, range_y = x_max - x_min, y_max - y_min

        # Auto-grid generation (~5 m resolution)
        if nx is None and ny is None:
            nx = max(30, int(np.ceil(range_x / 5.0)))
            ny = max(30, int(np.ceil(range_y / 5.0)))
        elif nx is None: nx = ny
        elif ny is None: ny = nx

        # Build observers array
        x_range = np.linspace(x_min, x_max, nx)
        y_range = np.linspace(y_min, y_max, ny)
        XX, YY = np.meshgrid(x_range, y_range, indexing='ij')

        observers = np.column_stack([XX.ravel(), YY.ravel(), np.full(nx * ny, z_slice)])
        
        # Check distances to avoid singularities
        self.check_observers_distances(observers, min_distance=0.1)

        # Compute pressure
        Nfreqs = len(self.Freqs)
        Nobs = observers.shape[0]
        print(f"\nComputing XY slice (z={z_slice:.2f} m, grid={nx}x{ny}), total observers={Nobs}...")
        
        p = self.acoustic_solver.compute_pressure(observers, block_size)

        if self.debug: print("Slice XY Pressure Norm: ", np.linalg.norm(p))

        # Reshape and Save data
        p = p.reshape((Nfreqs, nx, ny))
        
        self.acoustic_data["Freqs"]          = self.Freqs
        self.acoustic_data["P_sliceXY"]      = p
        self.acoustic_data["X_sliceXY"]      = x_range
        self.acoustic_data["Y_sliceXY"]      = y_range
        self.acoustic_data["Z_sliceXY"]      = z_slice
        self.acoustic_data["Obs_slicexy"]    = observers
        self.acoustic_data["Center_sliceXY"] = center

        self.save_acoustics()
        print(f"\n --> Slice XY data saved at {self._save_path}")

        self.acoustic_data.clear(); gc.collect()

        return self

    def run_sliceXZ(self,
                    y_slice   : float      = None,    # [m] Fixed y coordinate
                    nx        : int        = None,    # [-] Number of points in x-direction
                    nz        : int        = None,    # [-] Number of points in z-direction
                    xlim      : list       = None,    # [m] x limits [x_min, x_max] relative to center
                    zlim      : list       = None,    # [m] z limits [z_min, z_max] 
                    center    : np.ndarray = None,    # [m] Coordinates of the center in z-plane shape (2,)
                    margin    : float      = 10.0,    # [m] Absolute margin added to auto x limits
                    block_size: int        = 256):     # [-] Number of observers to process simultaneously

        self.check_acoustic_solver()

        if center is None: center = np.mean([t.AxisPos for t in self.turbines], axis=0)[:2]
        cx, cy = center[0], center[1]
        
        if y_slice is None: y_slice = cy

        # Extract all nodes X coordinates for auto-limits
        all_nodes_x = np.hstack([t.x[:, 0] for t in self.turbines])

        # Calculate bounding box
        if xlim is None:
            x_min, x_max = all_nodes_x.min() - margin, all_nodes_x.max() + margin
        else:
            x_min, x_max = cx + xlim[0], cx + xlim[1]

        if zlim is None:
            z_min, z_max = -self.Depth, 0.0
        else:
            z_min = max(zlim[0], -self.Depth)  # Ensure it doesn't go below seabed
            z_max = min(zlim[1], 0.0)          # Ensure it doesn't go above surface

        range_x, range_z = x_max - x_min, z_max - z_min

        # Auto-grid generation (~5 m resolution for X, ~2 m for Z)
        if nx is None and nz is None:
            nx = max(30, int(np.ceil(range_x / 5.0)))
            nz = max(30, int(np.ceil(range_z / 2.0)))
        elif nx is None: nx = nz
        elif nz is None: nz = nx

        # Build observers array
        x_range = np.linspace(x_min, x_max, nx)
        z_range = np.linspace(z_min, z_max, nz)
        XX, ZZ = np.meshgrid(x_range, z_range, indexing='ij')

        observers = np.column_stack([XX.ravel(), np.full(nx * nz, y_slice), ZZ.ravel()])

        # Check distances to avoid singularities
        self.check_observers_distances(observers, min_distance=0.1)

        # Compute pressure
        Nfreqs = len(self.Freqs)
        Nobs = observers.shape[0]
        print(f"\nComputing XZ slice (y={y_slice:.2f} m, grid={nx}x{nz}), total observers={Nobs}...")

        p = self.acoustic_solver.compute_pressure(observers, block_size)

        if self.debug: print("Slice XZ Pressure Norm: ", np.linalg.norm(p))

        # Reshape and Save data
        p = p.reshape((Nfreqs, nx, nz))

        self.acoustic_data["Freqs"]          = self.Freqs
        self.acoustic_data["P_sliceXZ"]      = p
        self.acoustic_data["X_sliceXZ"]      = x_range
        self.acoustic_data["Z_sliceXZ"]      = z_range
        self.acoustic_data["Y_sliceXZ"]      = y_slice
        self.acoustic_data["Obs_slicexz"]    = observers
        self.acoustic_data["Center_sliceXZ"] = center

        self.save_acoustics()
        print(f"\n --> Slice XZ data saved at {self._save_path}")

        self.acoustic_data.clear(); gc.collect()

        return self

    def run_sliceVertical(self,
                          azimuth   : float      = 0.0,     # [deg] Azimuth of the plane's normal (0° -> normal to +x)
                          center    : np.ndarray = None,    # [m] Coordinates of the center in XY shape (2,)
                          width     : float      = None,    # [m] Full span of the plane along its horizontal axis
                          z_range   : list       = None,    # [m] Depth extent [z_min, z_max]
                          nu        : int        = None,    # [-] Number of points along the span axis
                          nz        : int        = None,    # [-] Number of points along z-axis
                          margin    : float      = 10.0,    # [m] Absolute margin added to auto width
                          block_size: int        = 256):     # [-] Number of observers to process simultaneously
        """
        Computes pressure on a vertical plane whose normal points in an arbitrary
        horizontal direction defined by an azimuth angle.
        """

        self.check_acoustic_solver()

        # 1. Geometry: normal and span unit vectors in XY
        az_rad = np.deg2rad(azimuth)
        normal = np.array([ np.cos(az_rad),  np.sin(az_rad)])   # Normal to the plane
        span   = np.array([-np.sin(az_rad),  np.cos(az_rad)])   # In-plane horizontal axis

        # 2. Centre
        if center is None: center = np.mean([t.AxisPos for t in self.turbines], axis=0)[:2]
        cx, cy = center[0], center[1]

        # 3. Depth range
        if z_range is None:
            z_min, z_max = -self.Depth, 0.0
        else:
            z_min = max(z_range[0], -self.Depth)
            z_max = min(z_range[1], 0.0)

        # 4. Width: project all nodes onto the span axis
        if width is None:
            all_nodes_xy = np.vstack([t.x for t in self.turbines])[:, :2]
            rel_xy       = all_nodes_xy - np.array([cx, cy])
            proj_span    = rel_xy @ span
            
            u_min = proj_span.min() - margin
            u_max = proj_span.max() + margin
        else:
            u_min = -width / 2.0
            u_max =  width / 2.0

        range_u = u_max - u_min
        range_z = z_max - z_min

        # 5. Grid sizing (~5 m resolution for U, ~2 m for Z)
        if nu is None: nu = max(30, int(np.ceil(range_u / 5.0)))
        if nz is None: nz = max(30, int(np.ceil(range_z / 2.0)))

        # 6. Build observer grid in 3D
        u_vals = np.linspace(u_min, u_max, nu)
        z_vals = np.linspace(z_min, z_max, nz)
        
        UU, ZZ = np.meshgrid(u_vals, z_vals, indexing='ij')

        observers = np.column_stack([
            cx + UU.ravel() * span[0],
            cy + UU.ravel() * span[1],
            ZZ.ravel()
        ])

        # Check distances to avoid singularities
        self.check_observers_distances(observers, min_distance=0.1)

        # 7. Compute pressure
        Nfreqs = len(self.Freqs)
        Nobs   = observers.shape[0]
        
        print(f"\nComputing vertical slice (azimuth={azimuth:.1f}°, grid={nu}x{nz}), total observers={Nobs}...")
        print(f"  normal=({normal[0]:.2f}, {normal[1]:.2f}), span=({span[0]:.2f}, {span[1]:.2f})")
        
        p = self.acoustic_solver.compute_pressure(observers, block_size)

        if self.debug: print("Slice Vertical Pressure Norm: ", np.linalg.norm(p))

        # 8. Reshape and Save
        p      = p.reshape((Nfreqs, nu, nz))
        coords = observers.reshape((nu, nz, 3))

        self.acoustic_data["Freqs"]           = self.Freqs
        self.acoustic_data["P_sliceVertical"] = p
        self.acoustic_data["Coords_sliceV"]   = coords
        self.acoustic_data["U_sliceV"]        = u_vals
        self.acoustic_data["Z_sliceV"]        = z_vals
        self.acoustic_data["Azimuth_sliceV"]  = azimuth
        self.acoustic_data["Center_sliceV"]   = center

        self.save_acoustics()
        print(f"\n --> Vertical slice data saved at {self._save_path}")

        self.acoustic_data.clear(); gc.collect()

        return self

    def run_spheres(self,
                    r         : float = 30.,       # [m] Base radius (will be increased automatically if too small)
                    n_theta   : int   = 72,        # [-] Number of points in azimuthal direction
                    nz        : int   = 20,        # [-] Number of vertical layers
                    block_size: int   = 128):      # [-] Number of observers to process simultaneously
        """
        Computes the free-field pressure (N_images=0) on a spherical surface
        enclosing EACH turbine in the wind farm separately.
        
        For each turbine, the sphere is centered at its geometric center (BariPos/AxisPos).
        If the specified radius 'r' is too small to enclose all nodes with a 1m margin, 
        it is automatically increased for that specific turbine.
        """

        self.check_acoustic_solver()
        
        N_turbines = len(self.turbines)
        Nf         = len(self.Freqs)
        
        # Pre-allocate arrays to store results for ALL turbines
        P_all       = np.zeros((N_turbines, Nf, n_theta, nz), dtype=complex)
        Obs_all     = np.zeros((N_turbines, n_theta, nz, 3))
        Centers_all = np.zeros((N_turbines, 3))
        Radii_all   = np.zeros(N_turbines)
        
        # Temporarily disable boundary images for free-field radiation
        self.acoustic_solver.set_N_images(0)
        
        print(f"\nComputing free-field radiation spheres for {N_turbines} turbines...")
        
        for i, t in enumerate(self.turbines):
            # 1. Define the center for this turbine
            # Tries to use BariPos, falls back to AxisPos if not defined
            cx = getattr(t, 'BariPos', getattr(t, 'AxisPos', [0,0]))[0]
            cy = getattr(t, 'BariPos', getattr(t, 'AxisPos', [0,0]))[1]
            cz = float(np.mean(t.x[:, 2]))
            
            center = np.array([cx, cy, cz])
            
            # 2. Calculate minimum radius to enclose all nodes + 2.0 m margin
            node_dist = np.linalg.norm(t.x - center, axis=1)
            r_min     = node_dist.max() + 2.0
            
            # Enforce the minimum radius requirement
            r_actual = max(r, r_min)
            if r_actual > r:
                print(f"  Turbine {i+1}/{N_turbines}: Radius increased to {r_actual:.2f} m (Node max dist: {r_min-2.0:.2f} m).")
            else:
                print(f"  Turbine {i+1}/{N_turbines}: Radius {r_actual:.2f} m.")
                
            Centers_all[i] = center
            Radii_all[i]   = r_actual
            
            # 3. Create observer mesh (Vectorized)
            z_edges       = np.linspace(cz - r_actual, cz + r_actual, nz + 1)
            z_centers     = 0.5 * (z_edges[:-1] + z_edges[1:])
            
            theta_edges   = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
            theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
            
            Theta, Z = np.meshgrid(theta_centers, z_centers, indexing='ij')
            
            # Clip at 0.0 to prevent negative values inside sqrt due to float inaccuracies at the poles
            R_xy = np.sqrt(np.clip(r_actual**2 - (Z - cz)**2, 0.0, None))
            
            observers = np.column_stack([
                cx + (R_xy * np.cos(Theta)).ravel(),
                cy + (R_xy * np.sin(Theta)).ravel(),
                Z.ravel()
            ])
            
            # Ensure the margin logic worked correctly
            self.check_observers_distances(observers, min_distance=1.0)
            
            # 4. Compute pressure for this specific turbine's sphere
            p = self.acoustic_solver.compute_pressure(observers, block_size)
            
            if self.debug: 
                print(f"  -> Sphere {i+1} Pressure Norm: {np.linalg.norm(p):.2e}")
            
            # 5. Reshape and store
            P_all[i]   = p.reshape((Nf, n_theta, nz))
            Obs_all[i] = observers.reshape((n_theta, nz, 3))
            
        # Restore original solver settings (re-enabling images/reflections)
        self.acoustic_solver.restore_default_images()
        
        # Calculate approximate surface element area per sphere
        dA = (4.0 * np.pi * Radii_all**2) / (n_theta * nz)
        
        # Save aggregated data
        self.acoustic_data["Freqs"]           = self.Freqs
        self.acoustic_data["P_spheres"]       = P_all
        self.acoustic_data["Obs_spheres"]     = Obs_all
        self.acoustic_data["Centers_spheres"] = Centers_all
        self.acoustic_data["Radii_spheres"]   = Radii_all
        self.acoustic_data["N_theta"]         = n_theta
        self.acoustic_data["Nz_sphere"]       = nz
        self.acoustic_data["dA_spheres"]      = dA

        self.save_acoustics()
        print(f"\n --> Spheres data saved at {self._save_path}")

        self.acoustic_data.clear(); gc.collect()

        return self

    def run_all(self, 
                # Banderas de activación (On/Off)
                run_spectrums     : bool = True, 
                run_line          : bool = True, 
                run_polar         : bool = True, 
                run_cylinder      : bool = True, 
                run_sliceXY       : bool = True, 
                run_sliceXZ       : bool = True, 
                run_sliceVertical : bool = True, 
                run_spheres       : bool = True,
                
                # Diccionarios de configuración para sobreescribir los defaults
                kwargs_spectrums     : dict = None,
                kwargs_line          : dict = None,
                kwargs_polar         : dict = None,
                kwargs_cylinder      : dict = None,
                kwargs_sliceXY       : dict = None,
                kwargs_sliceXZ       : dict = None,
                kwargs_sliceVertical : dict = None,
                kwargs_spheres       : dict = None):
        """
        Runs a complete suite of acoustic evaluations.
        
        The execution order is sorted by default computational cost (number of observers),
        leaving the most intensive calculations (like 2D grids and 3D spheres) for the end.
        
        Users can pass dictionaries to configure specific parameters for each function.
        Example:
            farm.run_all(kwargs_sliceXY={"nx": 50, "ny": 50, "z_slice": -10.0})
        """
        
        print("\n" + "="*50)
        print("STARTING FULL ACOUSTIC SUITE")
        print("="*50)

        if run_spectrums:
            args = kwargs_spectrums if kwargs_spectrums is not None else {}
            self.run_spectrums(**args)

        if run_line:
            args = kwargs_line if kwargs_line is not None else {}
            self.run_line(**args)

        if run_polar:
            args = kwargs_polar if kwargs_polar is not None else {}
            self.run_polar(**args)

        if run_cylinder:
            args = kwargs_cylinder if kwargs_cylinder is not None else {}
            self.run_cylinder(**args)

        if run_sliceXY:
            args = kwargs_sliceXY if kwargs_sliceXY is not None else {}
            self.run_sliceXY(**args)

        if run_sliceXZ:
            args = kwargs_sliceXZ if kwargs_sliceXZ is not None else {}
            self.run_sliceXZ(**args)

        if run_sliceVertical:
            args = kwargs_sliceVertical if kwargs_sliceVertical is not None else {}
            self.run_sliceVertical(**args)

        if run_spheres:
            args = kwargs_spheres if kwargs_spheres is not None else {}
            self.run_spheres(**args)

        print("\n" + "="*50)
        print("FULL ACOUSTIC SUITE COMPLETED SUCCESSFULLY")
        print("="*50 + "\n")

        return self



