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
        self.cluster = cluster
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

                    if (idx+1) % 100 == 0: print(f"  Progress: {idx + 1}/{no}")

                    total_pressure[:, idx] += p_turb

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



