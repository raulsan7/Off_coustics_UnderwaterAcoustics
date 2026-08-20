"""
Module: AcousticMethods.py
Description: Implementation of different acoustic methods to solve acoustics in
             a z-bounded domain

Author: Raul Sanz Ramirez (raul.sanz.ramirez@upm.es / raul.sanz.ramirez@gmail.com)
Institution: Universidad Politecnica de Madrid - ETSIAE
Date: 07/2026 
"""

import gc
import abc
import numpy as np
import scipy.special as sp


class AcousticSolver(abc.ABC):
    """
    Abstract base class for all numerical acoustic methods.
    """

    @abc.abstractmethod
    def compute_pressure(self, observers, block_size):
        """
        Abstract base class for all numerical acoustic methods.
        """
        pass

    @abc.abstractmethod
    def get_name(self) -> str:
        pass


# ---------- Mehtod of Images ---------- #
class MethodImages(AcousticSolver):
    """
    Image Method solver for single WindTurbine or WindFarm
    """

    # ========== CONSTRUCTOR ========== #
    def __init__(self,
                 system,                            # [-] WindTurbine or WindFarm can be inputed
                 N_images         : int   = 30,     # [-] Number of image levels to compute
                 c_wat            : float = 1500.0, # [m/s] Speed of sound in fluid. Default: water --> 1500
                 rho_wat          : float = 1025.,  # [kg/m^3] Fluid density. Default: water --> 1000
                 Upper_BC         : int   = -1,     # [-] Upper Boundary condition -1 for  p = 0
                 Lower_BC         : int   = 1,      # [-] Lower Boundary condition +1 for dp = 0
                 Upper_HBC        : float = 0.,     # [m] z-coordinate of the upper reflecting plane (e.g. free surface)
                 Lower_HBC        : float = None,   # [m] z-coordinate of the lower reflecting plane (e.g. seabed)
                 attenuation_upper: float = 1.,     # [-] Attenuation coefficient for upper boundary reflections
                 attenuation_lower: float = 0.5,    # [-] Attenuation coefficient for lower boundary reflections
                 eps              : float = 1e-12,  # [-] Regularization term to avoid division by zero in distances
                 p_ref            : float = 1e-6,   # [Pa] Reference pressure
                 cluster          : bool  = False): # [-] Wheter the simulation is ran in a cluster or not (mor RAM if true)

        # Parse input
        self.N_images  = N_images
        self.c_wat     = c_wat
        self.rho_wat   = rho_wat
        self.Upper_BC  = Upper_BC
        self.Lower_BC  = Lower_BC
        self.Upper_HBC = Upper_HBC
        self.Lower_HBC = Lower_HBC
        self.up_R      = attenuation_upper
        self.lw_R      = attenuation_lower
        self.eps       = eps
        self.p_ref     = p_ref
        self.cluster   = cluster
        self._default_N_images = N_images

        if Lower_HBC is None or Upper_HBC is None or Upper_HBC <= Lower_HBC:
            raise ValueError(f"MethodImages requires Upper_HBC({Upper_HBC}) > Lower_HBC ({Lower_HBC})")

        # If system has atribute 'turbines' is a WindFarm class and qe use it
        # Else we assume a single WindTurbine and convert it to list
        if hasattr(system, 'turbines'):
            self.turbines = system.turbines
        elif isinstance(system, list):
            self.turbines = system
        else:
            self.turbines = [system]

        # Precompute impedance correction
        self.corrected_F = {}
        for turbine in self.turbines:
            self.corrected_F[turbine] = turbine.get_impedance_corrected_force(self.c_wat)

        # Precompute System of Images
        self._build_all_image_system()
            

    # ========== HELPERS ========== #
    def get_name(self) -> str:
        """
        Return a human-readable name of the solver.

        Returns
        -------
        str
        """

        return "Images Method"

    def _build_all_image_system(self):

        self.image_systems = {}
        for turbine in self.turbines:
            self.image_systems[turbine] = self._build_image_system(turbine.x, self.corrected_F[turbine])

        return self
    
    def _build_image_system(self,
                            nodes_pos_real: np.ndarray = None,      # [m] Coordinates of all dipolar sources
                            force         : np.ndarray = None):     # [N] Force array af all dipolar sources
        """
        Precomputes full 3D image geometry, forces, and boundary conditions.
        """

        # Ensure shape. For this method has to be shape(Nfreqs, 3, Nnodes)
        if force.shape[2] == 3: force = force.transpose(0,2,1)

        # Extract z coordinate for reflections
        zi = nodes_pos_real[:,2]

        # Compute reflections tree
        z_all, Force_out, parent, BC_all = self.method_of_images(zi, force)

        # Direct reconstruction of all image coordinates shape(3, Nnodes_total)
        nodes_pos = np.empty((3, z_all.size))
        nodes_pos[0,:] = nodes_pos_real[parent, 0]
        nodes_pos[1,:] = nodes_pos_real[parent, 1]
        nodes_pos[2,:] = z_all      

        return nodes_pos, Force_out, BC_all

    def set_N_images(self, 
                     new_N_images: int = None):  # [-] New number of images to set
        """
        Dynamically changes the number of images and recomputes the geometry.
        Useful for calculating direct fields (N_images=0)
        """
        self.N_images = new_N_images
        self._build_all_image_system()

        return self

    def restore_default_images(self):
        """
        Restores the image system to its original state defined at initialization
        """
        if self.N_images != self._default_N_images:
            self.N_images = self._default_N_images
            self._build_all_image_system()

        return self
    
    # ========== COMPUTE PRESSURE ========== #
    def compute_pressure(self,
                         observers : np.ndarray = None,        # [m] Observers coordinates array shape(:,3)
                         block_size: int        = 256):        # [-] Number of observers to process simultaneously
        """
        Compute acoustic pressure at observer positions for a given source system.

        Returns
        -------
        np.ndarray, shape (n_freq, n_obs)
            Complex pressure at each frequency and observer.
        """

        if observers is None: raise ValueError("MethodImages.compute_pressure(): observers cannot be None")

        ndim = observers.ndim
        coords = observers.shape[-1]

        if ndim != 2 or coords != 3:
            raise ValueError("MethodImages.compute_pressure(): observer must have shape(Nobservers, 3)")

        # Initialize pressure matrix
        nf = len(self.turbines[0].Freqs)
        no = len(observers)
        total_pressure = np.zeros((nf, no), dtype=complex)

        if not self.cluster: block_size = 1

        # Linear superposition
        if self.cluster:
            for nturb, turbine in enumerate(self.turbines):

                if len(self.turbines) > 1: print(f"\nTurbine {nturb+1}/{len(self.turbines)}: ")

                # Extract precomputed geometry
                nodes_pos, force, BC_all = self.image_systems[turbine]
                total_blocks = int(np.ceil(no / block_size))

                for start_idx in range(0, no, block_size):
                    end_idx = min(start_idx + block_size, no)
                    obs_block = observers[start_idx:end_idx]

                    # Compute pressure for the entire block at once
                    p_block = self.dipole_pressure_images(obs_block, turbine.Freqs, nodes_pos, force, BC_all)
                    total_pressure[:, start_idx:end_idx] += p_block

                    current_block = (start_idx // block_size) + 1
                    print(f"  Progress: Block {current_block}/{total_blocks} ({end_idx}/{no} observers)")
        else:
            for nturb, turbine in enumerate(self.turbines):

                if len(self.turbines) > 1: print(f"\nTurbine {nturb+1}/{len(self.turbines)}: ")

                # Extract precomputed geometry
                nodes_pos, force, BC_all = self.image_systems[turbine]

                for idx, obs in enumerate(observers):
                    p_turb =self.dipole_pressure_images(obs, turbine.Freqs, nodes_pos, force, BC_all)

                    total_pressure[:, idx] += p_turb[:,0]

                    if (idx+1) % 100 == 0: print(f"  Progress: {idx + 1}/{no}")

        return total_pressure


    # ========== METHOD ========== #
    def method_of_images(self, 
                         zi   : np.ndarray = None,      # [m] Vertical coordinates of real dipole nodess shape(Nnodes,)
                         Force: np.ndarray = None):     # [N] Force array shape(nfreqs, Nnodes, 3)
        """
        Apply the method of images for dipole forces with horizontal planar
        boundary conditions (reflections only along the z-direction).

        Returns
        -------
        z_all : ndarray, shape (Nnodes * (1 + 2*N),)
            z-coordinates of all nodes (real and image), ordered by blocks
            per real node.

        Force_out : ndarray, shape (Nfreqs, 3, Nnodes * (1 + 2*N))
            Forces at all nodes, including sign changes, z-reflections,
            and attenuation.

        is_real : ndarray of bool, shape (Nnodes * (1 + 2*N),)
            True for real nodes, False for image nodes.

        parent : ndarray of int, shape (Nnodes * (1 + 2*N),)
            Index of the original real node from which each node originates.
        """
    
        zi = np.asarray(zi, dtype=float)
        Nnodes = zi.size
        Nfreqs = Force.shape[0]
        N                    = self.N_images

        total_nodes = Nnodes*(1+2*N)

        z_all     = np.zeros(total_nodes)
        Force_out = np.zeros((Nfreqs, 3, total_nodes), dtype=Force.dtype)
        parent    = np.zeros(total_nodes, dtype=int)
        BC_all    = np.ones (total_nodes, dtype=int)

        idx = 0            # Global node counter
        for i_node in range(Nnodes):
            
            # --- Real Node --- #
            z_all[idx]         = zi[i_node]
            Force_out[:,:,idx] = Force[:,:,i_node]
            BC_all[idx]        = 1
            parent[idx]        = i_node

            idx += 1

            # Initial references
            z_upper = zi[i_node]
            z_lower = zi[i_node]

            F_upper = Force[:, :, i_node].copy()
            F_lower = Force[:, :, i_node].copy()

            last_plane_upper = "upper"
            last_plane_lower = "lower"

            for _ in range(1, N + 1):

                # ---------- upper chain ----------
                if last_plane_upper == "upper":
                    z_upper = 2.0 * self.Lower_HBC - z_upper
                    BC_u = self.Lower_BC
                    last_plane_upper = "lower"
                else:
                    z_upper = 2.0 * self.Upper_HBC - z_upper
                    BC_u = self.Upper_BC
                    last_plane_upper = "upper"

                F_upper[:, 2] *= -1.0
                F_upper *= self.up_R

                z_all[idx] = z_upper
                Force_out[:, :, idx] = F_upper
                BC_all[idx] = BC_u
                parent[idx] = i_node
                idx += 1

                # ---------- lower chain ----------
                if last_plane_lower == "lower":
                    z_lower = 2.0 * self.Upper_HBC - z_lower
                    BC_l = self.Upper_BC
                    last_plane_lower = "upper"
                else:
                    z_lower = 2.0 * self.Lower_HBC - z_lower
                    BC_l = self.Lower_BC
                    last_plane_lower = "lower"

                F_lower[:, 2] *= -1.0
                F_lower *= self.lw_R

                z_all[idx] = z_lower
                Force_out[:, :, idx] = F_lower
                BC_all[idx] = BC_l
                parent[idx] = i_node
                idx += 1


        return z_all, Force_out, parent, BC_all

    def dipole_pressure_images(self, 
                               observer_pos  : np.ndarray = None,   # [m] Observer coordinates shape(Nchunk,3)
                               freq          : np.ndarray = None,   # [Hz] Frequency array shape(nfreqs,)
                               nodes_pos     : np.ndarray = None,   # [m] Coordinates array for all dipole nodes shape(Nnodes_total, 3)
                               force         : np.ndarray = None,   # [N] Force array shape(nfreqs, Nnodes, 3)
                               BC_all        : np.ndarray = None):  # [-] Array with all boundary conditions

        observer_pos = np.asarray(observer_pos, dtype=float)
        if observer_pos.ndim == 1:
            observer_pos = observer_pos[np.newaxis, :]
        elif observer_pos.ndim != 2 or observer_pos.shape[1] != 3:
            raise ValueError("MethodImages.dipole_pressure_images(): observer_pos must have shape (Nobs, 3) or (3,)")

        # Distance vectors from each node to observer shape(Nchunk, 3, Nnodes_total)
        r_vec = observer_pos[:, :, np.newaxis] - nodes_pos[np.newaxis, :, :]

        # Scalar distance shape(Nchunk, Nnodes_total)
        r = np.linalg.norm(r_vec, axis=1)
        r = np.where(r<self.eps, self.eps, r)       # Avoid division by 0

        # Wavenumber shape(Nfreqs)
        k = 2 * np.pi * freq / self.c_wat

        # Project force across propagation direction shape(Nfeqs, Nnodes_total)
        F_dor_r = np.einsum('fjd, ojd -> fod', force, r_vec) / r[np.newaxis, :, :]
        del r_vec

        # Dipolar term from Green shape(Nfreqs, Nchunk, Nnodes_total)
        term1 = -1j * k[:, np.newaxis, np.newaxis] + 1./r[np.newaxis, :, :]

        # Phase + Green shape(Nfreqs, Nchunk, Nnodes_total)
        green = np.exp(1j * k[:, np.newaxis, np.newaxis] * r[np.newaxis, :, :]) / (4. * np.pi * r[np.newaxis, :, :])

        # Pressure per node shape(Nfreqs, Nchunk, Nnodes_total)
        p_nodes = F_dor_r * term1 * green * BC_all[np.newaxis, np.newaxis, :]

        # Sum all nodes contribution
        return np.sum(p_nodes, axis=2)


# ---------- Analytical Normal Modes ---------- #
class AnalyticalNormalModes(AcousticSolver):
    """
    Analytical Normal Modes solver for single WindTurbine or WindFarm
    """

        # ========== CONSTRUCTOR ========== #
    def __init__(self,
                 system,                            # [-] WindTurbine or WindFarm can be inputed
                 Nmodes           : int   = None,   # [-] Number of normal modes to retain
                 c_wat            : float = 1500.0, # [m/s] Speed of sound in fluid. Default: water --> 1500
                 rho_wat          : float = 1025.,  # [kg/m^3] Fluid density. Default: water --> 1000
                 Upper_HBC        : float = 0.,     # [m] z-coordinate of the upper reflecting plane (e.g. free surface)
                 Lower_HBC        : float = None,   # [m] z-coordinate of the lower reflecting plane (e.g. seabed)
                 eps              : float = 1e-12,  # [-] Regularization term to avoid division by zero in distances
                 p_ref            : float = 1e-6,   # [Pa] Reference pressure
                 cluster          : bool  = False,  # [-] Wheter the simulation is ran in a cluster or not (mor RAM if true)
                 verbose          : bool  = True):  # [-] Flag to print more info

        # Parse input
        self.m         = Nmodes
        self.c_wat     = c_wat
        self.rho_wat   = rho_wat
        self.Upper_HBC = Upper_HBC
        self.Lower_HBC = Lower_HBC
        self.eps       = eps
        self.p_ref     = p_ref
        self.cluster   = cluster
        self.verbose   = verbose

        if Lower_HBC is None or Upper_HBC is None or Upper_HBC <= Lower_HBC:
            raise ValueError(f"AnalyticalNormalModes requires Upper_HBC({Upper_HBC}) > Lower_HBC ({Lower_HBC})")
        else:
            self.H = np.abs(self.Lower_HBC-self.Upper_HBC)

        # If system has atribute 'turbines' is a WindFarm class and qe use it
        # Else we assume a single WindTurbine and convert it to list
        if hasattr(system, 'turbines'):
            self.turbines = system.turbines
        elif isinstance(system, list):
            self.turbines = system
        else:
            self.turbines = [system]

        # Ensure all turbines have matching depth
        if any(turbine.Depth != self.H for turbine in self.turbines):
            raise ValueError("Inputed boundary conditions do not match system's depth for one or more turbines")

        # Compute number of modes to retain
        if self.m is not None and self.m < 0:
            raise ValueError(f"AnalyticalNormalModes requires m = {self.m} > 0")
        
        if self.m is None:
            f_max = np.max(self.turbines[0].Freqs)
            m_prop = int(np.floor(0.5 * (4. * self.H * f_max / self.c_wat + 1)))    # Strictly propagating modes for f_max
            near_field_buffer = 5   # Margin for nearfield
            self.m = int(max(1, m_prop + near_field_buffer))

        # Precompute impedance correction
        # self.corrected_F = {}
        # for turbine in self.turbines:
        #     self.corrected_F[turbine] = turbine.get_impedance_corrected_force(self.c_wat)

        # Display mode information table
        if self.verbose:
            self._print_mode_summary()
  

    # ========== HELPERS ========== #
    def get_name(self) -> str:
        """
        Return a human-readable name of the solver.

        Returns
        -------
        str
        """

        return "Analytical Normal Modes"

    def _print_mode_summary(self):
        """
        Prints a formatted summary table of the retained normal modes, 
        their cutoff frequencies, and propagation behavior.
        """

        freqs = self.turbines[0].Freqs
        f_min, f_max = np.min(freqs), np.max(freqs)
        
        # Evaluate frequency-dependent variables at f_max
        k_medium = 2 * np.pi * f_max / self.c_wat

        line_width = 115
        print("\n" + "=" * line_width)
        print(f"{'ANALYTICAL NORMAL MODES SETUP':^{line_width}}")
        print("=" * line_width)
        print(f"  Water Depth (H):       {self.H:.2f} m")
        print(f"  Speed of Sound (c):    {self.c_wat:.2f} m/s")
        print(f"  Frequency Band:        {f_min:.2f} Hz - {f_max:.2f} Hz")
        print(f"  Retained Modes (m):    {self.m}")
        print(f"  * Note: k_rm and Horiz. Wavelength are evaluated at f_max ({f_max:.2f} Hz)")
        print("-" * line_width)
        print(f"  {'Mode':<5} | {'k_zm [rad/m]':<18} | {'Vert. WL [m]':<12} | {'f_cutoff [Hz]':<13} | {'k_rm [rad/m]':<20} | {'Horiz. WL [m]':<13} | {'Status'}")
        print("-" * line_width)

        if self.m <= 20:
            modes_to_show = list(range(1, self.m + 1))
        else:
            modes_to_show = list(range(1, 11)) + [-1] + list(range(self.m - 4, self.m + 1))

        for m_idx in modes_to_show:
            if m_idx == -1:
                print(f"  {'...':<5} | {'...':<18} | {'...':<12} | {'...':<13} | {'...':<20} | {'...':<13} | {'...'}")
                continue

            # Vertical properties (Frequency independent for hard bottom/surface)
            k_zm_val = (2 * m_idx - 1) * np.pi / (2. * self.H)
            k_zm_cplx = f"{k_zm_val:.4f}+0.0000j"
            lambda_zm = 4. * self.H / (2 * m_idx - 1)
            
            f_c = (2 * m_idx - 1) * self.c_wat / (4. * self.H)

            # Horizontal properties (Frequency dependent, evaluated at f_max)
            kr_squared = k_medium**2 - k_zm_val**2
            
            if kr_squared >= 0:
                k_rm_val = np.sqrt(kr_squared)
                k_rm_cplx = f"{k_rm_val:.4f}+0.0000j"
                lambda_rm = f"{2 * np.pi / k_rm_val:.2f}"
                status = "Propagating"
            else:
                k_rm_val = np.sqrt(np.abs(kr_squared))
                k_rm_cplx = f"0.0000+{k_rm_val:.4f}j"
                lambda_rm = "N/A (Decays)"
                status = "Evanescent"

            print(f"  {m_idx:<5d} | {k_zm_cplx:<18} | {lambda_zm:<12.2f} | {f_c:<13.2f} | {k_rm_cplx:<20} | {lambda_rm:<13} | {status}")

        print("=" * line_width + "\n")

    # ========== COMPUTE PRESSURE ========== #
    def compute_pressure(self,
                         observers : np.ndarray = None,        # [m] Observers coordinates array shape(:,3)
                         block_size: int        = 256):        # [-] Number of observers to process simultaneously
        """
        Compute acoustic pressure at observer positions for a given source system.

        Returns
        -------
        np.ndarray, shape (n_freq, n_obs)
            Complex pressure at each frequency and observer.
        """

        if observers is None: raise ValueError("AnalyticalNormalModes.compute_pressure(): observers cannot be None")

        ndim = observers.ndim
        coords = observers.shape[-1]

        if ndim != 2 or coords != 3:
            raise ValueError("AnalyticalNormalModes.compute_pressure(): observer must have shape(Nobservers, 3)")
        
        # Initialize pressure matrix
        nf = len(self.turbines[0].Freqs)
        no = len(observers)
        total_pressure = np.zeros((nf, no), dtype=complex)
        
        if not self.cluster: block_size = 1

        # Linear superposition
        if self.cluster:
            for nturb, turbine in enumerate(self.turbines):
        
                if len(self.turbines) > 1: print(f"\nTurbine {nturb+1}/{len(self.turbines)}: ")
                total_blocks = int(np.ceil(no / block_size))
        
                for start_idx in range(0, no, block_size):
                    end_idx = min(start_idx + block_size, no)
                    obs_block = observers[start_idx:end_idx]
        
                    # Compute pressure for the entire block at once
                    p_block = self.dipolar_pressure_NM(obs_block, turbine.Freqs, turbine.x, turbine.F)
                    total_pressure[:, start_idx:end_idx] += p_block
        
                    current_block = (start_idx // block_size) + 1
                    print(f"  Progress: Block {current_block}/{total_blocks} ({end_idx}/{no} observers)")
        else:
            for nturb, turbine in enumerate(self.turbines):
        
                if len(self.turbines) > 1: print(f"\nTurbine {nturb+1}/{len(self.turbines)}: ")
        
                for idx, obs in enumerate(observers):
                    p_turb = self.dipolar_pressure_NM(obs[np.newaxis,:], turbine.Freqs, turbine.x, turbine.F)
        
                    total_pressure[:, idx] += p_turb[:,0]
        
                    if (idx+1) % 100 == 0: print(f"  Progress: {idx + 1}/{no}")
        
        return total_pressure
        

    # ========== METHOD ========== #
    def Psi_m(self, 
              m: int   = None,         # [-] Mode number
              z: float = None):        # [m] Vertical coordinate of the observer
        """
        Computes analytical normal mode Psi_m(z) for a certain depth z. 
        Harcoded to Psi_m(0) = 0; dPsi_m(H) = 0.
        """

        return np.sqrt(2.*self.rho_wat/self.H) * np.sin((2*m-1)*np.pi*z/(2.*self.H))

    def dPsi_m_dz(self,
                  m: int = None,            # [-] Mode number
                  z: np.ndarray = None):    # [m] Vertical coordinate array
        """
        Computes the analytical derivative of the normal mode dPsi_m(z)/dz.
        """
        k_zm = (2*m-1)*np.pi/(2.*self.H)
        return np.sqrt(2.*self.rho_wat/self.H) * k_zm * np.cos(k_zm*z)

    def dipolar_pressure_NM(self,
                            observer_pos  : np.ndarray = None,   # [m] Observer coordinates shape(Nchunk,3)
                            freq          : np.ndarray = None,   # [Hz] Frequency array shape(nfreqs,)
                            nodes_pos     : np.ndarray = None,   # [m] Coordinates array for all dipole nodes shape(Nnodes_total, 3)
                            force         : np.ndarray = None):  # [N] Force array shape(nfreqs, Nnodes, 3)

        observer_pos = np.asarray(observer_pos, dtype=float)

        # Coordinate differences and distances
        R_vec = observer_pos[:, np.newaxis, :2] - nodes_pos[np.newaxis, :, :2]

        X, Y = R_vec[..., 0], R_vec[..., 1]

        # Horizontal distance
        R = np.linalg.norm(R_vec, axis=2)
        R = np.where(R<self.eps, self.eps, R)       # Avoid division by 0

        # Wavenumbers
        k = 2* np.pi * freq / self.c_wat

        # Preparation for calculations
        Nobs = observer_pos.shape[0]
        Nnodes = nodes_pos.shape[0]
        Nfreqs = len(freq)
        p_nodes = np.zeros((Nfreqs, Nobs, Nnodes), dtype=complex)

        # Dipolar force components shape(Nfreqs, Nnodes)
        Dx, Dy, Dz = force[:,:,0], force[:,:,1], force[:,:,2]
        z_obs = observer_pos[:,2]
        zs    = nodes_pos[:,2]

        # Superposition of modes
        for m in range(1, self.m+1):

            k_zm = (2*m-1)*np.pi/(2.*self.H)

            # Horizontal wavenumber
            k_rm = np.sqrt((k+0j)**2 - k_zm**2)     # shape(Nfreqs)

            # Modal argument for Hankel functions
            k_rm_R = k_rm[:, np.newaxis, np.newaxis] * R[np.newaxis, :, :]

            # Hankel functions
            H0 = sp.hankel1(0, k_rm_R)
            H1 = sp.hankel1(1, k_rm_R)

            # Normal modes evaluations
            psi_z   = self.Psi_m(m, z_obs)      # shape(Nobs,)
            psi_zs  = self.Psi_m(m, zs)         # shape(Nnodes,)
            dpsi_zs = self.dPsi_m_dz(m, zs)     # shape(Nnodes,)

            # Component z (dipole depth effect)
            term_z = Dz[:, np.newaxis, :] * (dpsi_zs[np.newaxis, np.newaxis, :] / self.rho_wat) * H0

            # Components x,y (dipole horizontal directivity effect)
            dx_part = Dx[:, np.newaxis, :] * (X[np.newaxis, :, :]/R[np.newaxis, :, :])
            dy_part = Dy[:, np.newaxis, :] * (Y[np.newaxis, :, :]/R[np.newaxis, :, :])

            term_xy = (psi_zs[np.newaxis, np.newaxis, :] * k_rm[:, np.newaxis, np.newaxis] /self.rho_wat) * H1 * (dx_part + dy_part)

            # Acumulate contribution for mode m
            p_m = psi_z[np.newaxis, :, np.newaxis] * (term_z + term_xy)
            p_nodes += p_m

        # Sum over all dipole sources and multiply by global coefficient
        return (-1j/4.) * np.sum(p_nodes, axis=2)








