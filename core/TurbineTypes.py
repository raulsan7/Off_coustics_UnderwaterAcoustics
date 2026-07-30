"""
Module: TurbineTypes.py
Description: Concrete implementations of specific offshore wind turbine types.
             Includes physical modeling, structural mass distributions, and 
             drivetrain vibration excitation spectra.

             ┌────────────────────────────────────────────────────────┐
             │                  TurbineTypes.py Class                 │
             │                                                        │
             │                   [ DTU10MWMonopile ]                  │
             │                            │                           │
             │        ┌───────────────────┴───────────────────┐       │
             │        ▼                                       ▼       │
             │  [ compute_mass ]                     [ compute_force ]│
             │        │                                       │       │
             │        ▼                                       ▼       │
             │  Analytical Thickness                 Spectral Forces  │
             │  & Added Mass Profiles                 via Drivetrain  │
             └────────────────────────────────────────────────────────┘

Author: Raul Sanz Ramirez (raul.sanz.ramirez@upm.es / raul.sanz.ramirez@gmail.com)
Institution: Universidad Politecnica de Madrid - ETSIAE
Date: 07/2026 
"""

import numpy as np
from pathlib import Path
from core.WindTurbine import WindTurbine



# ---------- TUBRINE MODEL 1 --------- #
class DTU10MWMonopile(WindTurbine):
    """
    Concrete implementation of the DTU 10 MW reference wind turbine on a monopile substructure.

    Key Properties:
    - D (float)        : Base monopile outer diameter (9.0 m).
    - rho_wat (float)  : Seawater density (1000.0 kg/m³).
    - rho_mat (float)  : Substructure steel material density (8500.0 kg/m³).
    - _path_rpm (Path) : File path to the rotor speed curve CSV.
    """

    # ========== CONSTRUCTOR ========== #
    def __init__(self, *args, **kwargs):
        """
        Initializes the DTU 10 MW Monopile turbine instance and verifies external resources.

        Inherits:
            Base properties and memory optimization (__slots__) from WindTurbine.

        Raises:
            FileNotFoundError: If the drivetrain RPM curve database file does not exist.
        """

        super().__init__(*args, **kwargs)

        self.D = 9.0            # [m] Turbine diameter
        self.rho_wat  = 1025.0   # [kg/m^3] Water density 
        self.rho_mat  = 8500.0   # [kg/m^3] Material density
        self.wet_area = 848.12   # [m^2] Wetted area

        # Hardcode where data should be located
        self._path_rpm = Path.cwd().resolve() / "wind_speed_curves_DTU_10MW" / "rpm_ws.csv"

        if not self._path_rpm.exists():
            raise FileNotFoundError(f"RPM curve file not found: {self._path_rpm}")
        
        self.case_type = "Monopile"

    # ========== COMPUTE SOURCE TERM ========== #
    def compute_force(self,
                      filter_freqs: bool = False,   # Wheter to skip non inputed frequencis in OpenFAST
                      verbose     : bool = True,    # Flag to print more info
                      skipf       : int = 1):       # Skips frequency data e.g. Freqs[::skipf]
        """
        Computes the frequency-domain acoustic dipole excitation forces.

        Execution Pipeline:
        ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
        │  Remove Nodes:   │ ──> │   Extract Wet:   │ ──> │ Mass Estimation: │
        │    Duplicates    │     │    z <= 0.0 m    │     │  m_eff = m + ma  │
        └──────────────────┘     └──────────────────┘     └──────────────────┘
                                                                   │
        ┌──────────────────┐     ┌──────────────────┐              │
        │ Force Correction │ <── │  RFFT Analysis:  │ <────────────┘
        │   (Hankel/Wind)  │     │   Time ──> Freq  │
        └──────────────────┘     └──────────────────┘

        Arguments:
        - filter_freqs [bool]  : Filter non-excited drivetrain frequencies if True.
        - verbose      [bool]  : Print memory consumption and structural parameters.
        - c_wat        [float] : Speed of sound in water (default: 1500 m/s).
        - skipf        [int]   : Frequency downsampling interval.

        Returns:
        - self (DTU10MWMonopile) : Updates self.Freqs, self.F, and coordinates in-place.
        """
        
        from utils.MathUtils import remove_duplicate_nodes, compute_rfft

        Nm = self.Nmembers
        Nn = self.Nnodes
        nt = self.Time.size
        dt = self.Time[1] - self.Time[0]

        # Remove duplicate nodes shared between consecutive members
        is_floating = True
        self.mask_duplicate = remove_duplicate_nodes(self.x_all, delete_last= is_floating)

        self.x = self.x_all.reshape((Nm * Nn, 3))[self.mask_duplicate, :]
        self.acc = self.acc.reshape((nt, Nm * Nn, 3))[:, self.mask_duplicate, :]

        # ---------- Remove dry nodes ---------- #
        wet_nodes = self.x[:,2] <= 0.0          # Surface is at z = 0
        self.x = self.x[wet_nodes, :]
        self.acc = self.acc[:, wet_nodes, :]
        Nnodes_wet = np.sum(wet_nodes)

        # ---------- Compute mass properties ---------- #
        mass, added_mass = self.compute_mass()              # This way mass is Harcoded to DTU10MW
        mass_effective = mass + added_mass                  # shape (Nnodes_wet,)
        del mass, added_mass  

        # ---------- Compute force via FFT ---------- #
        A, freqs = compute_rfft(self.acc, nt, dt, skipf=skipf, remove_zero=True)
        self.Freqs = freqs
        self.F     = - A * mass_effective[np.newaxis, :, np.newaxis]
        del freqs, mass_effective, A, self.acc

        # ---------- Filter Frequencies ---------- #
        if filter_freqs:
            mask_freqs_to_use = self.filter_frequencies()

            self.Freqs = self.Freqs[mask_freqs_to_use]
            self.F     = self.F[mask_freqs_to_use,:,:]

        # ---------- SUMMARY PRINT ---------- #
        Nfreqs = len(self.Freqs)
        F_memory_mb = self.F.nbytes / 1024. / 1024.
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"FORCE COMPUTATION COMPLETED")
            print(f"{'─'*68}")
            print(f"Nodes (wet): {Nnodes_wet:6d}")
            print(f"Frequencies: {Nfreqs:6d}")
            print(f"Memory (F):  {F_memory_mb:6.2f} MB")
            print(f"{'='*70}\n")

        return self

    def compute_mass(self):
        """
        Calculates dry structural mass and hydrodynamic added mass distributions.

        Thickness profile used for the DTU 10 MW Monopile (based on elevation z):
        ┌────────────────────────────────────────────────────────┐
        │ Elevation range (z)     │ Inner diameter (D_inner)     │
        ├─────────────────────────┼──────────────────────────────┤
        │ [ 4.0 , 10.0 ] m        │ 8.70 m                       │
        │ [-10.0 ,  4.0 ] m        │ 8.69 m                       │
        │ [ < -10.0    ] m        │ 8.80 m                       │
        └────────────────────────────────────────────────────────┘

        Formulas Applied:
        - Structural Area : A_mass = pi * (R_outer^2 - R_inner^2)
        - Fluid Area      : A_added = pi * R_outer^2

        Returns:
        - mass       [ndarray] (kg) : Distributed dry mass per submerged node.
        - added_mass [ndarray] (kg) : Hydrodynamic added mass per submerged node.
        """
        
        from utils.MathUtils import divide_span

        D       = self.D
        Depth   = self.Depth
        rho_mat = self.rho_mat
        rho_wat = self.rho_wat

        Nnodes  = self.x.shape[0]

        z_nodes = np.linspace(-Depth, 0, Nnodes)
        ds      = divide_span(z_nodes)

        D_outer = np.full(len(z_nodes), D)
        D_inner = np.zeros_like(D_outer)

        # Build thickness array
        cond = (z_nodes >= 4.0) & (z_nodes <= 10.0)
        D_inner[cond] = 8.7
        
        cond = (z_nodes >= -10) & (z_nodes<4.0)
        D_inner[cond] = 8.69

        cond = z_nodes < -10.0
        D_inner[cond] = 8.80

        A_mass       = np.pi * ((D_outer / 2.0)**2 - (D_inner / 2.0)**2)
        A_added_mass = np.pi * (D_outer / 2.0)**2

        mass       = A_mass       * rho_mat * ds
        added_mass = A_added_mass * rho_wat * ds

        return mass, added_mass

    def filter_frequencies(self):
        """
        Calculates the frequency mask corresponding to physical drivetrain excitations.

        Steps:
        1. Read the RPM curve at the current WindSpeed.
        2. Generate the corresponding drivetrain spectrum peaks (shaft & gear meshes).
        3. Build a frequency band mask to clean numerical noise below 10.0 Hz.

        Returns:
        - mask_freqs_to_use [ndarray] (bool) : Boolean mask to filter active frequencies.
        """

        from utils.IOUtils import read_curve
        from utils.MathUtils import generate_timeseries_banded_sines, filter_non_usefull_freqs
        
        freqs = self.Freqs
        rpm   = read_curve(self._path_rpm)(self.WindSpeed)
        freqs_amp, keys = drivetrain10MW_excitation_spectrum(rpm, alpha_mesh=0.5)
        _, freqs_to_use = generate_timeseries_banded_sines(freqs_amp, keys, self.Time, used_freqs=True)
        mask_freqs_to_use = filter_non_usefull_freqs(freqs, freqs_to_use, freqs_over=10.0)
        print("DTU10MWMonopile.filter_frequencies(): FREQS_OVER IS HARCODED TO 10.0 Hz")

        return mask_freqs_to_use

    def get_impedance_corrected_force(self,
                             c_wat: float = 1500):  # [m/s] Speed of sound in fluid. Default: water --> 1500
        
        from utils.MathUtils import alpha_hankel

        omega = 2* np.pi * self.Freqs                       # [rad/s] Angular frequency
        k     = omega / c_wat                               # [1/m]   Wavenumber
        alpha = alpha_hankel(k, self.D)
        corrected_force = self.F * np.abs(alpha[:, np.newaxis, np.newaxis])  # [N] Corrected force shape (Nfreqs, Nnodes_wet, 3)

        return corrected_force
# ------------------------------------ #



# ---------- TUBRINE MODEL 2 --------- #
class DTU10MWFloating(WindTurbine):
    """
    Concrete implementation of the DTU 10 MW wind turbine on a floating platform.

    Platform Design:
    - Dimensioned and optimized by CENER (Centro Nacional de Energías Renovables).
    - Engineered for deep-water dynamic stability and coupled aero-hydro-servo-elastic response.

    Key Properties:
    - D (float)        : Nominal column/shaft diameter [m].
    - rho_wat (float)  : Seawater density (1000.0 kg/m³).
    - rho_mat (float)  : Structural steel material density (8500.0 kg/m³).
    - Platform (str)   : CENER floating platform reference design.
    - _path_rpm (Path) : File path to the rotor speed curve CSV.
    """

    # ========== CONSTRUCTOR ========== #
    def __init__(self, *args, **kwargs):
        """
        Initializes the DTU 10 MW Floating turbine instance and verifies external resources.

        Inherits:
            Base properties and memory optimization (__slots__) from WindTurbine.

        Raises:
            FileNotFoundError: If the drivetrain RPM curve database file does not exist.
        """

        super().__init__(*args, **kwargs)

        self.col_D    = 14.5     # [m] Comlumns diameter
        self.Xseca    = 10.875   # [m] Pontoon width
        self.Xsecb    = 7.0      # [m] Pontoon height
        self.rho_wat  = 1025.0   # [kg/m^3] Water density 
        self.rho_mat  = 7850.0   # [kg/m^3] Material density
        self.t        = 0.023    # [m] Wall thickness
        self.wet_area = 9359.07   # [m^2] Wetted area
        
        self.col_members = [[0,1],[2,3],[4,5]]  # [-] Column members ID list
        self.pon_members = [[6],[7],[8]]        # [-] Pontoon members ID lists

        # Hardcode where data should be located
        self._path_rpm = Path.cwd().resolve() / "wind_speed_curves_DTU_10MW" / "rpm_ws.csv"

        if not self._path_rpm.exists():
            raise FileNotFoundError(f"RPM curve file not found: {self._path_rpm}")
        
        self.case_type = "Floating"


    # ========== COMPUTE SOURCE TERM ========== #
    def compute_force(self,
                      filter_freqs: bool = False,   # Wheter to skip non inputed frequencis in OpenFAST
                      verbose     : bool = True,    # Flag to print more info
                      skipf       : int = 1):       # Skips frequency data e.g. Freqs[::skipf]
        """
        Computes the frequency-domain acoustic dipole excitation forces for the floating platform.

        Execution Pipeline:
        ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
        │  Acceleration    │ ──> │ Mass Estimation: │ ──> │ Force Engine     │
        │   Time ──> Freq  │     │ Broadcast Col/Pon│     │ $F = -A \\cdot m$ │
        └──────────────────┘     └──────────────────┘     └──────────────────┘
                                                                   │
        ┌──────────────────┐     ┌──────────────────┐              │
        │ Dual-Geometry    │ <── │ Remove Duplicate │ <────────────┘
        │ Hankel Correction│     │   & Dry Nodes    │
        └──────────────────┘     └──────────────────┘

        Substructure Mapping & Correction:
        - Columns  : Corrected using cylindrical Hankel factors with outer diameter `col_D`.
        - Pontoons : Corrected via equivalent cylinder mapping using square cross-section area.

        Arguments:
        - filter_freqs [bool]  : Filter non-excited drivetrain frequencies if True.
        - verbose      [bool]  : Print memory consumption and wet node summary parameters.
        - c_wat        [float] : Speed of sound in water (default: 1500 m/s).
        - skipf        [int]   : Frequency downsampling interval.

        Returns:
        - self (DTU10MWFloating) : Updates self.Freqs, self.F, and coordinates in-place.
        """
        
        from utils.MathUtils import compute_rfft
        
        Nm = self.Nmembers
        Nn = self.Nnodes
        nt = self.Time.size
        dt = self.Time[1] - self.Time[0]

        col_member_ids = self.col_members
        pon_member_ids = self.pon_members

        # ---------- Acceleration FFT ---------- #
        self.acc             = self.acc.reshape((nt, Nm * Nn, 3))
        self.acc, self.Freqs = compute_rfft(self.acc, nt, dt, skipf=skipf, remove_zero=True)
        nf                   = len(self.Freqs)
        A                    = self.acc.reshape(nf, Nm, Nn, 3)

        # Free up memory
        self.acc =  None

        # ---------- Compute mass properties ---------- #
        # Harcoded to DTU10MW DeltaWnd platform
        (mass_col, added_mass_col, mass_pon, added_mass_pon,
         _, _) = self.compute_mass()

        mass_eff_col = mass_col + added_mass_col        # shape(2*Nn-1,)
        mass_eff_pon = mass_pon + added_mass_pon        # shape(Nn,)

        mass_eff = np.zeros((Nm, Nn))
        for col in col_member_ids:
            m0, m1 = col
            mass_eff[m0, :] = mass_eff_col[:Nn]
            mass_eff[m1, :] = mass_eff_col[Nn - 1:]

        for pon in pon_member_ids:
            m = pon[0]
            mass_eff[m, :] = mass_eff_pon

        # ---------- Compute force via F = m_eff * acc ---------- #
        self.F = - A * mass_eff[np.newaxis, :, :, np.newaxis]  # shape (Nfreqs, Nm, Nn, 3)

        # Free up memory
        A, mass_eff = None, None
        mass_eff_col, mass_col, added_mass_col = None, None, None
        mass_eff_pon, mass_pon, added_mass_pon = None, None, None

        # ---------- Filter Frequencies ---------- #
        if filter_freqs:
            mask_freqs_to_use = self.filter_frequencies()

            self.Freqs = self.Freqs[mask_freqs_to_use]
            self.F     = self.F[mask_freqs_to_use,:,:,:]

        nf         = len(self.Freqs)
        self.Time  = None

        # ---------- Remove duplicate nodes and dry nodes ---------- #
        keep = np.ones((Nm, Nn), dtype=bool)

        for col in col_member_ids:
            keep[col[1], 0] = False  # Remove first node of second member (shared node)
        
        keep[self.x_all[:,:,2]>0] = False  # Remove dry nodes (z > 0)
        keep_flat  = keep.reshape((Nm * Nn,))
        
        self.x = self.x_all.reshape((Nm * Nn, 3))[keep_flat, :]
        self.F     = self.F.reshape((nf, Nm * Nn, 3))[:, keep_flat, :]

        Nnodes_wet = self.x.shape[0]

        # To apply impedance correction later
        self.keep_flat = keep_flat

        # ---------- SUMMARY PRINT ---------- #
        Nfreqs = len(self.Freqs)
        F_memory_mb = self.F.nbytes / 1024. / 1024.
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"FORCE COMPUTATION COMPLETED")
            print(f"{'─'*68}")
            print(f"Nodes (wet): {Nnodes_wet:6d}")
            print(f"Frequencies: {Nfreqs:6d}")
            print(f"Memory (F):  {F_memory_mb:6.2f} MB")
            print(f"{'='*70}\n")

        return self

    def compute_mass(self):
        """
        Computes the structural mass and fluid added mass vectors for the CENER platform.

        Extracts multi-segment properties from one representative column and pontoon, 
        accounting for cross-section profiles and continuous arc-lengths.

        Substructure Geometries:
        - Cylindrical Columns (2 Members):
          A_struct = pi * (R_outer^2 - R_inner^2)
          A_added  = pi * R_outer^2
          *Note: Bottom/Top closure lids are explicitly added as localized steel masses (t).

        - Hollow Square Pontoons (1 Member):
          A_struct = (X_a * X_b) - (X_a - 2*t) * (X_b - 2*t)
          A_added  = X_a * X_b

        Returns:
        - mass_col       [ndarray] (2*Nn-1,) : Distributed column dry structural mass.
        - added_mass_col [ndarray] (2*Nn-1,) : Hydrodynamic column added mass.
        - mass_pon       [ndarray] (Nn,)      : Distributed pontoon dry structural mass.
        - added_mass_pon [ndarray] (Nn,)      : Hydrodynamic pontoon added mass.
        - types_col      [list of str]       : Structural identifiers for columns.
        - types_pon      [list of str]       : Structural identifiers for pontoons.
        """
        
        from utils.MathUtils import divide_span
        
        pos_0   = self.x_all       # (Nm, Nn, 3)
        Nn      = pos_0.shape[1]

        D       = self.col_D
        t       = self.t
        rho_mat = self.rho_mat
        rho_wat = self.rho_wat
        Xseca   = self.Xseca
        Xsecb   = self.Xsecb

        col_member_ids = self.col_members   # e.g. [[0,1], [2,3], [4,5]]
        pon_member_ids = self.pon_members   # e.g. [[6], [7], [8]]

        # ------------------------------------------------------------------ #
        # COLUMN  (representative: first column = col_member_ids[0])
        # ------------------------------------------------------------------ #
        # Build full z-coordinate vector for the column, removing shared node
        col_ids   = col_member_ids[0]           # e.g. [0, 1]
        col_nodes = [pos_0[m, :, :] for m in col_ids]  # list of (Nn, 3)

        # Concatenate dropping the first node of each subsequent member
        # (it is identical to the last node of the previous one)
        col_pos = np.concatenate([col_nodes[0]] + [seg[1:] for seg in col_nodes[1:]], axis=0)  # (2*Nn - 1, 3)
        Nnodes_col = col_pos.shape[0]

        length = np.linalg.norm(col_pos[-1, :] - col_pos[0, :])                   # (2*Nn - 1,)
        nodes_col_1d = np.linspace(0,length, Nnodes_col)
        ds_col = divide_span(nodes_col_1d)             # (2*Nn - 1,)  correct spans

        R_outer_col = D / 2.0
        R_inner_col = R_outer_col - t

        A_struct_col     = np.pi * (R_outer_col**2 - R_inner_col**2)
        A_added_col      = np.pi *  R_outer_col**2

        mass_col       = A_struct_col * rho_mat * ds_col        # (2*Nn - 1,)
        added_mass_col = A_added_col  * rho_wat * ds_col        # (2*Nn - 1,)

        mass_col[0]  += np.pi *  R_outer_col**2 * rho_mat * t    # Top lid mass
        mass_col[-1] += np.pi *  R_outer_col**2 * rho_mat * t    # Bottom lid mass


        # ------------------------------------------------------------------ #
        # PONTOON  (representative: first pontoon = pon_member_ids[0])
        # ------------------------------------------------------------------ #
        pon_id  = pon_member_ids[0][0]          # e.g. 6
        pon_pos = pos_0[pon_id, :, :]           # (Nn, 3)

        # Arc-length along the pontoon axis (may be horizontal)
        diffs      = np.diff(pon_pos, axis=0)                   # (Nn-1, 3)
        seg_len    = np.linalg.norm(diffs, axis=1)              # (Nn-1,)
        arc_length = np.concatenate([[0.0], np.cumsum(seg_len)])# (Nn,)
        ds_pon     = divide_span(arc_length)                    # (Nn,)

        A_struct_pon = Xseca * Xsecb - (Xseca-2*t) * (Xsecb-2*t)# hollow square cross-section
        A_added_pon  = Xseca * Xsecb                            # full square for added mass

        mass_pon       = A_struct_pon * rho_mat * ds_pon    # (Nn,)
        added_mass_pon = A_added_pon  * rho_wat * ds_pon    # (Nn,)

        struct_types_col = ["column"]  * len(col_member_ids)   # ["column", "column", "column"]
        struct_types_pon = ["pontoon"] * len(pon_member_ids)   # ["pontoon", "pontoon", "pontoon"]

        return (mass_col, added_mass_col, mass_pon, added_mass_pon, struct_types_col, struct_types_pon)

    def filter_frequencies(self):
        """
        Calculates the frequency mask corresponding to physical drivetrain excitations.

        Steps:
        1. Read the RPM curve at the current WindSpeed.
        2. Generate the corresponding drivetrain spectrum peaks (shaft & gear meshes).
        3. Build a frequency band mask to clean numerical noise below 10.0 Hz.

        Returns:
        - mask_freqs_to_use [ndarray] (bool) : Boolean mask to filter active frequencies.
        """

        from utils.IOUtils import read_curve
        from utils.MathUtils import generate_timeseries_banded_sines, filter_non_usefull_freqs
        
        freqs = self.Freqs
        rpm   = read_curve(self._path_rpm)(self.WindSpeed)
        freqs_amp, keys = drivetrain10MW_excitation_spectrum(rpm, alpha_mesh=0.5)
        _, freqs_to_use = generate_timeseries_banded_sines(freqs_amp, keys, self.Time, used_freqs=True)
        mask_freqs_to_use = filter_non_usefull_freqs(freqs, freqs_to_use, freqs_over=10.0)
        print("DTU10MWFloating.filter_frequencies(): FREQS_OVER IS HARCODED TO 10.0 Hz")

        return mask_freqs_to_use

    def get_impedance_corrected_force(self,
                             c_wat: float = 1500):  # [m/s] Speed of sound in fluid. Default: water --> 1500

        from utils.MathUtils import alpha_hankel

        Nm = self.Nmembers
        Nn = self.Nnodes
        Nnodes_wet = self.x.shape[0]
        nf = len(self.Freqs)
        col_member_ids = self.col_members
        pon_member_ids = self.pon_members
        keep_flat = self.keep_flat

        # ---------- Correct force source ---------- #
        original_to_wet = np.full(Nm * Nn, -1, dtype=int)
        original_to_wet[keep_flat] = np.arange(Nnodes_wet)

        
        alpha_full = np.ones((nf, Nnodes_wet), dtype=complex)

        omega = 2* np.pi * self.Freqs                       # [rad/s] Angular frequency
        k     = omega / c_wat                               # [1/m]   Wavenumber

        # Columns
        alpha_col = alpha_hankel(k, self.col_D)
        for col in col_member_ids:
            m0, m1 = col
            idx_flat = np.concatenate([
                np.arange(m0*Nn, m0*Nn +Nn),
                np.arange(m1*Nn + 1, m1*Nn + Nn),       # Skip duplicate node
            ])
            node_idx = original_to_wet[idx_flat]
            node_idx = node_idx[node_idx >= 0]          # Keep only wet nodes
            if len(node_idx) > 0:
                alpha_full[:, node_idx] = alpha_col[:, np.newaxis]

        # Pontoons
        D_pon = 2*np.sqrt(self.Xseca * self.Xsecb/np.pi)  # Equivalent diameter for pontoon cross-section
        alpha_pon = alpha_hankel(k, D_pon)
        for pon in pon_member_ids:
            m = pon[0]
            idx_flat = np.arange(m*Nn, m*Nn + Nn)
            node_idx = original_to_wet[idx_flat]
            node_idx = node_idx[node_idx >= 0]          # Keep only wet nodes
            if len(node_idx) > 0:
                alpha_full[:, node_idx] = alpha_pon[:, np.newaxis]
        
        corrected_force = self.F * np.abs(alpha_full[:, :, np.newaxis])  # [N] Corrected force shape (Nfreqs, Nnodes_wet, 3)

        alpha_full, original_to_wet, k, omega = None, None, None, None


        return corrected_force
# ------------------------------------ #



# ---------- TUBRINE MODEL 3 --------- #
class SAITEC2MWFloating(WindTurbine):
    """
    Concrete implementation of the SAITEC SENVION 2 MW floating wind turbine.

    Platform Concept (SATH Technology):
    • Uses a swing-around-single-point (SATH) dual-hull concrete platform.
    • Features twin horizontal ellipsoidal/elliptical concrete floaters.

    Substructure Layout:
    ┌────────────────────────────────────────────────────────┐
    │                    SAITEC DUAL HULL                    │
    │                                                        │
    │   [ Floater 1 (Left) ]  <═══  Rigid Joint  ═══>  [  Tower Base  ]
    │      Horizontal, Oval        (Virtual Links)            Center
    │   [ Floater 2 (Right)]  <═════════════════════>  [-8.95m, 0.0m]
    └────────────────────────────────────────────────────────┘

    Key Properties:
    • Diam_y, Diam_z [float] (m) : Elliptical cross-section diameters (10.0 m x 12.5 m).
    • rho_mat        [float]     : Concrete material density (1850.0 kg/m³).
    • float_members  [list]      : IDs mapping the twin horizontal concrete hulls.
    • joint_members  [list]      : Virtual rigid links/joints mapping structural connections.
    """

    # ========== CONSTRUCTOR ========== #
    def __init__(self, *args, **kwargs):
        """
        Initializes the SAITEC 2 MW Floating instance and loads rotational performance curves.

        Inherits:
            Base properties and geometry processing from WindTurbine.

        Notes:
            • Verifies the existence of Saitec-specific drivetrain curve files.
            • Overrides case type to "Floating Horizontal".
            • Expected baricenter coordinates: $(-8.9586, 0.0)$ m.

        Raises:
            FileNotFoundError: If the 2 MW RPM speed curve file is missing.
        """

        super().__init__(*args, **kwargs)

        self.rho_wat       = 1025.0         # [kg/m^3] Water density 
        self.rho_mat       = 1850.0         # [kg/m^3] Material density
        self.Diam_y        = 10.0           # [m] Column diameter in y-direction
        self.Diam_z        = 12.5           # [m] Column diameter in z-direction
        self.t             = 1.0            # [m] Wall thickness
        self.float_members = [[0,1],[3,4]]  # [-] Floaters members ID lists
        self.joint_members = [[5,2],[6]]    # [-] Virtual joint (rigid links) members ID lists
        self.wet_area      = None
        print("SAITEC2MWFloating.__init__(): CHECK IF BARIPOS IS WELL COMPUTED, SHOULD BE (-8.9586, 0)")

        # Hardcode where data should be located
        self._path_rpm = Path.cwd().resolve() / "wind_speed_curves_SAITEC_2MW" / "rpm_ws.csv"

        if not self._path_rpm.exists():
            raise FileNotFoundError(f"RPM curve file not found: {self._path_rpm}")

        self.case_type = "Floating Horizontal"


    # ========== COMPUTE SOURCE TERM ========== #
    def compute_force(self,
                      filter_freqs: bool = False,   # [-] Wheter to skip non inputed frequencis in OpenFAST
                      verbose     : bool = True,    # [-] Flag to print more info
                      skipf       : int = 1):       # [-] Skips frequency data e.g. Freqs[::skipf]
        """
        Computes the frequency-domain acoustic dipole excitation forces for Saitec floaters.

        Execution Pipeline:
        ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
        │  Acceleration    │ ──> │ Mass Extraction: │ ──> │ Frequency Filter │
        │   Time ──> Freq  │     │ Elliptical Hull  │     │  (Drivetrain)    │
        └──────────────────┘     └──────────────────┘     └──────────────────┘
                                                                   │
        ┌──────────────────┐     ┌──────────────────┐              │
        │ Elliptical Wave  │ <── │ Keep Wet & Non-  │ <────────────┘
        │ Hankel Correction│     │ Duplicate Nodes  │
        └──────────────────┘     └──────────────────┘

        Acoustic Scattering Correction:
        • Corrected using cylindrical Hankel factors with an equivalent circular diameter:
          $$D_{\text{eff}} = \\sqrt{D_y \\cdot D_z}$$

        Arguments:
        • filter_freqs [bool]  : Filter non-excited drivetrain frequencies if True.
        • verbose      [bool]  : Print memory consumption and wet node summary parameters.
        • c_wat        [float] : Speed of sound in water (default: 1500 m/s).
        • skipf        [int]   : Frequency downsampling interval.

        Returns:
        • self (SAITEC2MWFloating) : Updates self.Freqs, self.F, and coordinates in-place.
        """
        
        from utils.MathUtils import compute_rfft

        Nm = len(self.float_members)
        Nn = self.Nnodes
        nt = self.Time.size
        dt = self.Time[1] - self.Time[0]

        # ---------- Acceleration FFT ---------- #
        self.acc             = self.acc.reshape((nt, Nm*Nn, 3))
        self.acc, self.Freqs = compute_rfft(self.acc, nt, dt, skipf=skipf, remove_zero=True)
        nf                   = len(self.Freqs)
        A                    = self.acc.reshape((nf, Nm, Nn, 3))

        # Free up memory
        self.acc = None


        # ---------- Compute mass properties ---------- #
        mass_float, added_mass_float = self.compute_mass()
        mass_eff_float = mass_float + added_mass_float
        mass_float, added_mass_float = None, None

        # ---------- Filter Frequencies ---------- #
        if filter_freqs:
            mask_freqs_to_use = self.filter_frequencies()

            self.Freqs = self.Freqs[mask_freqs_to_use]
            A          = A[mask_freqs_to_use,:,:,:]

        nf         = len(self.Freqs)
        self.Time  = None

        # ---------- Remove duplicate and dry nodes ---------- #
        keep = np.ones((Nm,Nn), dtype=bool)

        float_members = sum(self.float_members, [])     # Linearize list
        self.x = self.x_all[float_members]

        # Remove duplicate node
        Nm_per_float = len(self.float_members[0])
        keep[(np.arange(Nm) % Nm_per_float) != 0, 0] = False

        # Remove dry nodes
        keep[self.x_all[:,:,2]>0] = False
        self.x = self.x_all[keep].reshape((-1,3))
        A      = A[:, keep, :].reshape((nf, -1, 3))

        mass_eff_float = np.concatenate([mass_eff_float, mass_eff_float])
        self.F = - A * mass_eff_float[np.newaxis,:,np.newaxis]
        A, mass_eff_float = None, None
        Nnodes_wet = self.x.shape[0]

        # ---------- SUMMARY PRINT ---------- #
        Nfreqs = len(self.Freqs)
        F_memory_mb = self.F.nbytes / 1024. / 1024.
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"FORCE COMPUTATION COMPLETED")
            print(f"{'─'*68}")
            print(f"Nodes (wet): {Nnodes_wet:6d}")
            print(f"Frequencies: {Nfreqs:6d}")
            print(f"Memory (F):  {F_memory_mb:6.2f} MB")
            print(f"{'='*70}\n")

        return self

    def compute_mass(self):
        """
        Calculates dry structural mass and hydrodynamic added mass for the concrete hulls.

        Mass calculations account for the elliptical hollow cross-section of the floaters 
        submerged under the operational draft (z_draft):

        Formulas Applied:
        • Outer Ellipse Area : A_outer = integral_elipse_z(D_y, D_z, -z_draft)
        • Inner Ellipse Area : A_inner = integral_elipse_z(D_y - 2*t, D_z - 2*t, -z_draft)
        • Hollow Shell Area  : A_shell = A_outer - A_inner

        Discretization:
        • Reconstructed continuously along the horizontal arc-length ($s$) of a single representative hull.
        • Symmetrically broadcasted across both concrete floaters.

        Returns:
        • mass       [ndarray] (2*Nn-1,) [kg] : Concrete dry structural mass distribution.
        • added_mass [ndarray] (2*Nn-1,) [kg] : Hydrodynamic added mass distribution.

        Raises:
            RuntimeError: If floater members are incorrectly defined or contain NaN coordinates.
        """
        
        from utils.MathUtils import divide_span, integral_elipse_z
        
        p     = self._p
        pos_0 = self.pos_0_raw

        t       = self.t
        rho_mat = self.rho_mat
        rho_wat = self.rho_wat

        float_members = self.float_members
        # Only one floater because the other is the same
        nodes_float = np.vstack([pos_0[float_members[0][0], :, :]] + [pos_0[m, 1:, :] for m in float_members[0][1:]])
        if np.isnan(np.asarray(nodes_float)).any():raise RuntimeError(f"Float members are not well assigned. {float_members} some member has a NaN")

        segment_lengths = np.linalg.norm(np.diff(nodes_float, axis=0), axis=1)
        s               = np.zeros(len(nodes_float))
        s[1:]           = np.cumsum(segment_lengths)
        ds_float        = divide_span(s)

        z_draft = nodes_float[0,-1]
        A_outer = integral_elipse_z(p["Diam_y"], p["Diam_z"], k=-z_draft)
        A_inner = integral_elipse_z(p["Diam_y"]-2*t, p["Diam_z"]-2*t, k=-z_draft)
        A_shell = A_outer-A_inner

        mass = A_shell * rho_mat * ds_float
        added_mass =  A_outer * rho_wat * ds_float

        # for i in range(len(nodes_float)):
        #     print(i, nodes_float[i][0], ds_float[i])

        return mass, added_mass     # shape (2*Nn-1, )

    def filter_frequencies(self):
        """
        Calculates the frequency mask corresponding to physical drivetrain excitations.

        Steps:
        1. Read the RPM curve at the current WindSpeed.
        2. Generate the corresponding drivetrain spectrum peaks (shaft & gear meshes).
        3. Build a frequency band mask to clean numerical noise below 10.0 Hz.

        Returns:
        - mask_freqs_to_use [ndarray] (bool) : Boolean mask to filter active frequencies.
        """

        from utils.IOUtils import read_curve
        from utils.MathUtils import generate_timeseries_banded_sines, filter_non_usefull_freqs
        
        freqs = self.Freqs
        rpm   = read_curve(self._path_rpm)(self.WindSpeed)
        freqs_amp, keys = drivetrain2MW_excitation_spectrum(rpm, alpha_mesh=0.5)
        _, freqs_to_use = generate_timeseries_banded_sines(freqs_amp, keys, self.Time, used_freqs=True)
        mask_freqs_to_use = filter_non_usefull_freqs(freqs, freqs_to_use, freqs_over=10.0)
        print("SAITEC2MWFloating.filter_frequencies(): FREQS_OVER IS HARCODED TO 10.0 Hz")

        return mask_freqs_to_use

    def get_impedance_corrected_force(self,
                             c_wat: float = 1500):  # [m/s] Speed of sound in fluid. Default: water --> 1500

        from utils.MathUtils import alpha_hankel

        omega = 2*np.pi*self.Freqs
        k     = omega / c_wat
        D_eff = np.sqrt(self.Diam_y*self.Diam_z)/2.

        alpha_floats = alpha_hankel(k, D_eff)
        corrected_force = self.F * np.abs(alpha_floats[:,np.newaxis, np.newaxis])

        self.pos_0_raw, k, omega = None, None, None

        return corrected_force
# ------------------------------------ #



###################################
# DRIVETRAIN EXCITATION FUNCTIONS #
###################################

def drivetrain10MW_excitation_spectrum(rpm       : float = 9.6,     # [rpm] Rotor Speed
                                       damping   : float = 0.02,    # [-] Structural damping
                                       p_shaft   : float = 1.5,     # [-] Saft harmonic decay
                                       alpha_mesh: float = 0.8):    # [-] Gear mesh harmonic decay
    """
    Construct physically-consistent excitation spectrum for the
    10 MW medium-speed drivetrain of Wang, Nejad & Moan.

    Frequencies are taken from:
        Wang, S., Nejad, A.R., Moan, T.
        "Design and Dynamic Analysis of a Compact 10 MW Medium Speed
        Gearbox for Offshore Wind Turbines"

    Torsional eigenfrequencies (modes 1-14) are taken directly from
    Table (Drivetrain torsional modes) of the paper.

    Shaft and gear-mesh excitation frequencies are interpolated
    between minimum rotor speed (6 rpm) and maximum rotor speed (9.6 rpm)
    using linear interpolation, consistent with proportional scaling
    of rotational frequencies.

    -------------------------------------------------------------
    AMPLITUDE MODELLING (must be cited if published)
    -------------------------------------------------------------

    1) Shaft harmonics decay:
       A_n ∝ 1 / n^p
       Justification:
       - Hansen, M.O.L., Aerodynamics of Wind Turbines
       - Burton et al., Wind Energy Handbook
       - Bossanyi (2003), Individual Pitch Control
       Typical p in [1,2]; here p = 1.5 (default).

    2) Gear mesh harmonic decay:
       A_m = A_1 exp(-alpha (m-1))
       Justification:
       - McFadden (1986, 1987)
       - Kahraman (1994)
       - Randall, Vibration-based Condition Monitoring
       Typical alpha in [0.5,1.5]; here alpha = 0.8 (default).

    3) Modal transfer function:
       Classical SDOF frequency response function:

       H(f) = sqrt( Σ_i 1 / [ (1 - (f/f_n)^2)^2
                              + (2 ζ f/f_n)^2 ] )

       Justification:
       - Inman, Engineering Vibration
       - Ewins, Modal Testing
       - Rao, Mechanical Vibrations

       ζ = 2% assumed structural damping (typical steel drivetrain).

    4) Final amplitude:
       A_total(f_k) = A_source(f_k) * H(f_k)

    5) Spectrum normalized to unit RMS:
       Σ (A_k^2 / 2) = 1

       Time series can later be scaled to desired force RMS.

    6) Base energy hierarchy weights:

        Relative base amplitudes between shafts and gearbox stages
        are assigned according to the observed torsional energy
        distribution in multi-stage wind turbine drivetrains:

        - Aerodynamic 3P dominates at LSS
        - Energy attenuates toward IMS and HSS
        - Gear mesh Stage 1 > Stage 2 > Stage 3

        Justification:
        - Peeters et al. (2006), Wind Energy
        - Nejad et al. (2016), Renewable Energy
        - Kahraman (1994), ASME Journal of Vibration and Acoustics
        - Guo & Parker (2012), Journal of Sound and Vibration

        These weights represent a physically consistent
        energy hierarchy rather than exact measured amplitudes.

    Parameters
    ----------
    rpm : float
        Rotor speed [rpm] (valid range 6-9.6)
    damping : float
        Modal damping ratio ζ
    p_shaft : float
        Polynomial decay exponent for shaft harmonics
    alpha_mesh : float
        Exponential decay factor for gear mesh harmonics

    Returns
    -------
    freqs_amp : ndarray (N,2)
        Column 0: frequency [Hz]
        Column 1: peak amplitude (unit RMS normalized)
    keys : list of str
        Label for each frequency component
    """

    # -----------------------------
    # Rotor speed limits
    # -----------------------------
    min_rpm = 6.0
    max_rpm = 9.6
    rpm = np.clip(rpm, min_rpm, max_rpm)

    # -----------------------------
    # Torsional eigenfrequencies (Hz)
    # -----------------------------
    torsional_modes = np.array([
        3.942, 16.683, 33.811, 61.736, 72.015,
        108.447, 175.792, 184.351, 214.403,
        245.476, 274.443, 336.874, 346.436, 391.730
    ])

    # -----------------------------
    # Shaft frequencies (min / max rpm)
    # -----------------------------
    shaft_min = np.array([
        0.1, 0.2, 0.3, 0.6,
        0.322, 0.644, 0.966,
        1.383, 2.766, 4.149,
        5.017, 10.034, 15.051
    ])

    shaft_max = np.array([
        0.160, 0.320, 0.480, 0.640,
        0.515, 1.030, 1.545,
        2.213, 4.426, 6.639,
        8.028, 16.056, 24.024
    ])

    shaft_keys = [
        "lss_1p","lss_2p","lss_3p","lss_6p",
        "ims1_1p","ims1_2p","ims1_3p",
        "ims2_1p","ims2_2p","ims2_3p",
        "hss_1p","hss_2p","hss_3p"
    ]

    # -----------------------------
    # Gear mesh frequencies

    # -----------------------------
    mesh_min = np.array([
        10.30, 20.6, 30.9,
        48.952, 97.904, 146.856,
        93.423, 186.846, 280.269
    ])

    mesh_max = np.array([
        16.48, 32.96, 49.44,
        78.3, 156.6, 234.9,
        149.492, 298.984, 448.476
    ])

    mesh_keys = [
        "lss_mesh_1p","lss_mesh_2p","lss_mesh_3p",
        "ims_mesh_1p","ims_mesh_2p","ims_mesh_3p",
        "hss_mesh_1p","hss_mesh_2p","hss_mesh_3p"
    ]
    

    # -----------------------------
    # Linear interpolation
    # -----------------------------
    factor = (rpm - min_rpm) / (max_rpm - min_rpm)

    shaft_freqs = shaft_min + factor * (shaft_max - shaft_min)
    mesh_freqs = mesh_min + factor * (mesh_max - mesh_min)

    freqs = np.concatenate([shaft_freqs, mesh_freqs])
    keys = shaft_keys + mesh_keys

    # -----------------------------
    # Base physical weights (energy hierarchy)
    # -----------------------------
    A_base = {

        # ----- LSS shaft -----
        "lss_1p": 0.6,
        "lss_2p": 0.3,
        "lss_3p": 1.0,
        "lss_6p": 0.2,

        # ----- IMS1 shaft -----
        "ims1_1p": 0.4,
        "ims1_2p": 0.2,
        "ims1_3p": 0.6,

        # ----- IMS2 shaft -----
        "ims2_1p": 0.3,
        "ims2_2p": 0.15,
        "ims2_3p": 0.4,

        # ----- HSS shaft -----
        "hss_1p": 0.2,
        "hss_2p": 0.1,
        "hss_3p": 0.3,

        # ----- Gear mesh stages -----
        "lss_mesh_1p": 0.8-0.0,
        "lss_mesh_2p": 0.8-0.0,
        "lss_mesh_3p": 0.8-0.0,

        "ims_mesh_1p": 0.5-0.0,
        "ims_mesh_2p": 0.5-0.0,
        "ims_mesh_3p": 0.5-0.0,

        "hss_mesh_1p": 0.3-0.0,
        "hss_mesh_2p": 0.3-0.0,
        "hss_mesh_3p": 0.3-0.0,
    }

    amps_source = []

    for key in keys:

        base = A_base[key]

        if "mesh" not in key:
            n = int(key.split("_")[-1].replace("p",""))
            A = base #* (1.0 / (n**p_shaft))

        else:
            m = int(key.split("_")[-1].replace("p",""))
            A = base * np.exp(-alpha_mesh*(m-1))

        amps_source.append(A)

    amps_source = np.array(amps_source)

    # -----------------------------
    # Modal transfer function
    # -----------------------------
    H = np.zeros_like(freqs)

    for i, f in enumerate(freqs):
        modal_sum = 0.0
        for fn in torsional_modes:
            modal_sum += 1.0 / (
                (1 - (f/fn)**2)**2 +
                (2*damping*f/fn)**2
            )
        H[i] = np.sqrt(modal_sum)

    A_total = amps_source * H

    # -----------------------------
    # Normalize to unit RMS
    # -----------------------------
    rms = np.sqrt(np.sum((A_total**2)/2))
    A_total /= rms

    freqs_amp = np.column_stack([freqs, A_total])

    return freqs_amp, keys

def drivetrain2MW_excitation_spectrum(rpm       : float = 17.1,     # [rpm] Rotor Speed
                                      path_barRB: str   = None,     # [-] Path to rotor bearing data
                                      path_barGB: str   = None,     # [-] Path to gearbox data
                                      fmin      : float = 0.0,      # [Hz] Minimum frequency
                                      fmax      : float = np.inf,   # [Hz] Maximum frequency
                                      tol       : float = 1e-3,     # [-] Frequency tolerance
                                      Normalised: bool  = True):    # [-] Flag to normalize spectrum
    """
    Construct excitation spectrum for the 2 MW variable-speed wind turbine
    described in Escaler & Mebarki (2018).

    Parameters
    ----------
    rpm : float
        Rotor speed [rpm] (typical range 9 - 14.9, rated = 14.5)
    path_barRB : str or None
        Path to CSV file with bar-chart data for rotor bearings.
    path_barGB : str or None
        Path to CSV file with bar-chart data for gearbox.
    fmin : float
        Lower frequency bound [Hz] (inclusive).
    fmax : float
        Upper frequency bound [Hz] (exclusive).
    tol : float
        Frequency tolerance [Hz] for merging duplicates.
        Two frequencies are considered equal if |f1-f2| < tol.

    Returns
    -------
    freqs_amp : ndarray, shape (N,2)
        Column 0: frequency [Hz]
        Column 1: normalised amplitude (unit RMS)
    keys : list of str
        Label for each frequency component (only the ones kept).
    """

    if path_barRB is None: path_barRB = "/home/hp/simulations/openfast_simulations/py_codes/wind_speed_curves_SAITEC_2MW/Bar_RB12.csv"
    if path_barGB is None: path_barGB = "/home/hp/simulations/openfast_simulations/py_codes/wind_speed_curves_SAITEC_2MW/Bar_GB123.csv"

    def _freq_to_label(f, f0):
        """Return a short label for a given frequency."""
        # Simple heuristic: compare to integer multiples of f0, fp, f1, f2, f3, gmfp, gmf12, gmf23
        # For exact matching we use the pre -defined keys from compute_freqs; this is only for fallback.
        return f"f_{f:.2f}Hz"

    def _compute_f3(rpm):
        """Return f3 (high -speed shaft frequency) for a given rotor speed."""
        f0 = rpm / 60.0
        zs, zr = 18, 87
        f1 = f0 * (zs + zr) / zs
        z12, z21 = 70, 16
        f2 = z12 / z21 * f1
        z23, z32 = 84, 19
        f3 = z23 / z32 * f2
        return f3

    def compute_freqs(f0_rpm, all_peaks=False):
        """
        Computes frequencies based on the 2 MW wind turbine kinematics.
        Same as the user -provided function.
        """
        f0 = f0_rpm / 60.0

        zs  = 18     # Sun
        zp  = 34     # Planet
        zr  = 87     # Ring
        z12 = 70     # Gear 1
        z21 = 16     # Gear 2
        z23 = 84     # Gear 3
        z32 = 19     # Gear 4

        f1 = f0 * (zs + zr) / zs
        fp = zs / zp * (f1 - f0)
        f2 = z12 / z21 * f1
        f3 = z23 / z32 * f2
        gmfp = zp * fp
        gmf12 = z12 * f1
        gmf23 = z23 * f2
        fb = 3 * f0

        if all_peaks:
            RB1 = np.array([f0, fb, 2*fb, 3*fb, 4*fb, gmfp, f3, 2*gmfp, 2*f3, 3*gmfp, 3*f3, 4*gmfp])
            RB2 = np.array([f0, fb,       3*fb, 4*fb, gmfp, f3, 2*gmfp, 2*f3, 3*gmfp, 3*f3, 4*gmfp])
            GB1 = np.array([gmfp, f3, 2*gmfp, 2*f3, 3*gmfp, 3*f3, 4*gmfp, 5*gmfp, 6*gmfp])
            GB2 = np.array([gmfp, f3, gmf12, 2*gmf12, 3*gmf12, 4*gmf12, 5*gmf12, 6*gmf12, gmf23])
            GB3 = np.array([gmf12, 2*gmf12, 3*gmf12, 4*gmf12, 5*gmf12, 6*gmf12, gmf23, 2*gmf23, 3*gmf23, 4*gmf23, 5*gmf23])
            return RB1, RB2, GB1, GB2, GB3
        else:
            return f0, f1, f2

    def input_bars(pathRB, pathGB, f0=17.1):
        """
        Reads bar -chart data from CSV files (same as user -provided function).
        """
        RB1f, RB2f, GB1f, GB2f, GB3f = compute_freqs(f0, all_peaks=True)

        data_RB = np.genfromtxt(pathRB, delimiter=',', skip_header=1)
        RB1 = data_RB[:, 0]; RB2 = data_RB[:,1]
        RB1 = RB1[~np.isnan(RB1)]
        RB2 = RB2[~np.isnan(RB2)]

        RB1 = np.vstack([RB1f, RB1]).transpose()
        RB2 = np.vstack([RB2f, RB2]).transpose()

        data_GB = np.genfromtxt(pathGB, delimiter=',', skip_header=1)
        GB1 = data_GB[:, 0]; GB2 = data_GB[:,1]; GB3 = data_GB[:,2]
        GB1 = GB1[~np.isnan(GB1)]
        GB2 = GB2[~np.isnan(GB2)]
        GB3 = GB3[~np.isnan(GB3)]

        GB1 = np.vstack([GB1f, GB1]).transpose()
        GB2 = np.vstack([GB2f, GB2]).transpose()
        GB3 = np.vstack([GB3f, GB3]).transpose()

        # Sort each array by frequency
        def sort_by_frequency(arr):
            return arr[arr[:,0].argsort()]

        RB1 = sort_by_frequency(RB1)
        RB2 = sort_by_frequency(RB2)
        GB1 = sort_by_frequency(GB1)
        GB2 = sort_by_frequency(GB2)
        GB3 = sort_by_frequency(GB3)

        return RB1, RB2, GB1, GB2, GB3


    # ------------------------------
    # 1. Compute all frequencies from the paper's kinematics
    # ------------------------------
    RB1f, RB2f, GB1f, GB2f, GB3f = compute_freqs(rpm, all_peaks=True)

    # Combine all frequencies with their source keys
    freqs_raw = []
    keys_raw = []

    for f in RB1f:
        freqs_raw.append(f)
        keys_raw.append('RB1_' + _freq_to_label(f, rpm/60))
    for f in RB2f:
        freqs_raw.append(f)
        keys_raw.append('RB2_' + _freq_to_label(f, rpm/60))
    for f in GB1f:
        freqs_raw.append(f)
        keys_raw.append('GB1_' + _freq_to_label(f, rpm/60))
    for f in GB2f:
        freqs_raw.append(f)
        keys_raw.append('GB2_' + _freq_to_label(f, rpm/60))
    for f in GB3f:
        freqs_raw.append(f)
        keys_raw.append('GB3_' + _freq_to_label(f, rpm/60))

    # Add electromagnetic peak at 84xf3 (observed in generator, Fig.7)
    f0 = rpm / 60.0
    f3 = _compute_f3(rpm)          # from gearbox kinematics
    f_em = 84 * f3
    freqs_raw.append(f_em)
    keys_raw.append('generator_84xf3')

    freqs_raw = np.array(freqs_raw)

    # ------------------------------
    # 2. Assign amplitudes
    # ------------------------------
    if path_barRB is not None and path_barGB is not None:
        # Read from external CSV files (same format as input_bars)
        RB1, RB2, GB1, GB2, GB3 = input_bars(path_barRB, path_barGB, f0=rpm)
        # Combine amplitudes in the same order as freqs_raw
        amps_raw = []
        # RB1
        for i in range(len(RB1f)):
            amps_raw.append(RB1[i,1])
        # RB2
        for i in range(len(RB2f)):
            amps_raw.append(RB2[i,1])
        # GB1
        for i in range(len(GB1f)):
            amps_raw.append(GB1[i,1])
        # GB2
        for i in range(len(GB2f)):
            amps_raw.append(GB2[i,1])
        # GB3
        for i in range(len(GB3f)):
            amps_raw.append(GB3[i,1])
        # Electromagnetic (no CSV entry, use default)
        amps_raw.append(0.2)
        amps_raw = np.array(amps_raw)
    else:
        raise ValueError("External bar -chart data for RB and GB must be provided via path_barRB and path_barGB.")

    # ------------------------------
    # 3. Sort by frequency
    # ------------------------------
    sort_idx = np.argsort(freqs_raw)
    freqs_sorted = freqs_raw[sort_idx]
    amps_sorted = amps_raw[sort_idx]
    keys_sorted = [keys_raw[i] for i in sort_idx]

    # ------------------------------
    # 4. Merge duplicate frequencies (keep larger amplitude)
    # ------------------------------
    merged_freqs = []
    merged_amps = []
    merged_keys = []

    i = 0
    while i < len(freqs_sorted):
        # Find all indices that are within tol of current frequency
        j = i
        while j+1 < len(freqs_sorted) and (freqs_sorted[j+1] - freqs_sorted[i]) < tol:
            j += 1
        # Among indices i..j, pick the one with largest amplitude
        best_idx = i + np.argmax(amps_sorted[i:j+1])
        merged_freqs.append(freqs_sorted[best_idx])
        merged_amps.append(amps_sorted[best_idx])
        merged_keys.append(keys_sorted[best_idx])
        i = j + 1

    # Convert to numpy arrays
    merged_freqs = np.array(merged_freqs)
    merged_amps = np.array(merged_amps)

    # ------------------------------
    # 5. Apply frequency range filter (fmin, fmax)
    # ------------------------------
    mask = (merged_freqs >= fmin) & (merged_freqs < fmax)
    freqs = merged_freqs[mask]
    amps = merged_amps[mask]
    keys = [merged_keys[i] for i in range(len(mask)) if mask[i]]

    # ------------------------------
    # 6. Normalise to unit RMS ( Σ (A²/2) = 1 )
    # ------------------------------
    if Normalised:
        if len(amps) > 0:
            rms = np.sqrt(np.sum((amps ** 2) / 2.0))
            amps /= rms
        else:
            # No components in the chosen range - return empty array
            freqs = np.array([])
            amps = np.array([])
            keys = []

    freqs_amp = np.column_stack([freqs, amps])

    return freqs_amp, keys


