"""
Module: AcousticMethods.py
Description: Implementation of different acoustic methods to solve acoustics in
             a z-bounded domain

Author: Raul Sanz Ramirez (raul.sanz.ramirez@upm.es / raul.sanz.ramirez@gmail.com)
Institution: Universidad Politecnica de Madrid - ETSIAE
Date: 07/2026 
"""

import abc
import numpy as np


class AcousticSolver(abc.ABC):
    """
    Abstract base class for all numerical acoustic methods.
    """

    @abc.abstractmethod
    def compute_pressure(self, system, observers):
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
                 p_ref            : float = 1e-6):

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

        if Lower_HBC is None or Upper_HBC is None or Upper_HBC <= Lower_HBC:
            raise ValueError(f"MethodImages requires Upper_HBC({Upper_HBC}) > Lower_HBC ({Lower_HBC})")


    # ========== HELPERS ========== #
    def get_name(self) -> str:
        """
        Return a human-readable name of the solver.

        Returns
        -------
        str
        """

        return "Images Method"
    
    # ========== COMPUTE PRESSURE ========== #
    def compute_pressure(self, 
                         system                  = None,        # [m] WindTurbine or WindFarm can be inputed
                         observers  : np.ndarray = None,        # [m] Observers coordinates array to compute pressure at shape(:,3)
                         print_every: int        = 10):         # [-] Print info every specific number of observers
        """
        Compute acoustic pressure at observer positions for a given source system.

        This method must be implemented by subclasses.

        Returns
        -------
        np.ndarray, shape (n_freq, n_obs)
            Complex pressure at each frequency and observer.
        """

        ndim = observers.ndim
        coords = observers.shape[-1]

        if ndim != 2 or coords != 3:
            raise ValueError("MethodImages.compute_pressure(): observer must have shape(Nobservers, 3)")

        # If system has atribute 'turbines' is a WindFarm class and qe use it
        # Else we assum a single WindTurbine and convert it to list
        turbines = getattr(system, 'turbines', [system])

        # Initialize pressure matrix
        first_turbine = turbines[0]
        nf = len(first_turbine.Freqs)
        no = len(observers)
        total_pressure = np.zeros((nf, no), dtype=complex)

        # Linear superposition
        for nturb, turbine in enumerate(turbines):

            p_turb = np.zeros_like(total_pressure)
            if len(turbines) > 1 : print(f"\nTurbine {nturb+1}/{len(turbines)}: ")

            Local_corrected_F = turbine.get_impedance_corrected_force(self.c_wat)    # Apply impedance correction to source
            for idx, obs in enumerate(observers):

                p_turb[:, idx] = self.dipole_pressure_images(turbine.x, obs, turbine.Freqs, Local_corrected_F)

                if (idx+1) % print_every == 0: 
                    print(f"  Progress: {idx + 1}/{no}")
                
            total_pressure += p_turb
        return total_pressure


    # ========== METHOD ========== #
    def method_of_images(self, 
                         zi   : np.ndarray = None,      # [m] Vertical coordinates of real dipole nodess shape(Nnodes,)
                         Force: np.ndarray = None):     # [N] Force array shape(nfreqs, Nnodes, 3)
        """
        Apply the method of images for dipole forces with horizontal planar
        boundary conditions (reflections only along the z-direction).

        Parameters
        ----------
        zi : array_like, shape (Nnodes,)
            z-coordinates of the real nodes.

        Force : ndarray, shape (Nfreqs, 3, Nnodes)
            Frequency-domain force vectors applied at the real nodes.

        Upper_BC, Lower_BC : int
            Boundary condition type at the upper and lower planes.
            +1 : Neumann-type (dF/dz = 0)
            -1 : Dirichlet-type (F = 0)

        Upper_HBC, Lower_HBC : float
            z-coordinates of the upper and lower boundary planes.

        N : int, optional
            Number of image layers. Each real node generates 2*N image nodes.

        attenuation : float, optional
            Attenuation factor applied at each reflection.

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
        
        Upper_BC, Lower_BC   = self.Upper_BC, self.Lower_BC
        Upper_HBC, Lower_HBC = self.Upper_HBC, self.Lower_HBC
        N                    = self.N_images
        attenuation_upper    = self.up_R
        attenuation_lower    = self.lw_R

        zi = np.asarray(zi, dtype=float)
        Nnodes = zi.size
        Nfreqs = Force.shape[0]

        total_nodes = Nnodes*(1+2*N)

        z_all     = np.zeros(total_nodes)
        Force_out = np.zeros((Nfreqs, 3, total_nodes), dtype=Force.dtype)
        is_real   = np.zeros(total_nodes, dtype=bool)
        parent    = np.zeros(total_nodes, dtype=int)
        BC_all    = np.ones (total_nodes, dtype=int)

        idx = 0            # Global node counter

        for i_node in range(Nnodes):
            
            # --- Real Node --- #
            z_all[idx]         = zi[i_node]
            Force_out[:,:,idx] = Force[:,:,i_node]
            BC_all[idx]        = 1
            is_real[idx]       = True
            parent[idx]        = i_node

            idx += 1

            # Initial references
            z_upper = zi[i_node]
            z_lower = zi[i_node]

            F_upper = Force[:, :, i_node]
            F_lower = Force[:, :, i_node]

            last_plane_upper = "upper"
            last_plane_lower = "lower"

            for level in range(1, N + 1):

                # ---------- upper chain ----------
                if last_plane_upper == "upper":
                    z_upper = 2.0 * Lower_HBC - z_upper
                    BC = Lower_BC
                    last_plane_upper = "lower"
                else:
                    z_upper = 2.0 * Upper_HBC - z_upper
                    BC = Upper_BC
                    last_plane_upper = "upper"

                F_upper = F_upper.copy()
                F_upper[:, 2] *= -1.0
                F_upper *= attenuation_upper

                z_all[idx] = z_upper
                Force_out[:, :, idx] = F_upper
                BC_all[idx] = BC
                is_real[idx] = False
                parent[idx] = i_node
                idx += 1

                # ---------- lower chain ----------
                if last_plane_lower == "lower":
                    z_lower = 2.0 * Upper_HBC - z_lower
                    BC = Upper_BC
                    last_plane_lower = "upper"
                else:
                    z_lower = 2.0 * Lower_HBC - z_lower
                    BC = Lower_BC
                    last_plane_lower = "lower"

                F_lower = F_lower.copy()
                F_lower[:, 2] *= -1.0
                F_lower *= attenuation_lower

                z_all[idx] = z_lower
                Force_out[:, :, idx] = F_lower
                BC_all[idx] = BC
                is_real[idx] = False
                parent[idx] = i_node
                idx += 1


        return z_all, Force_out, is_real, parent, BC_all

    def dipole_pressure_images(self, 
                               nodes_pos_real: np.ndarray = None,   # [m] Coordinates array for all dipole nodes shape(Nnodes, 3)
                               observer_pos  : np.ndarray = None,   # [m] Observer coordinates shape(3,)
                               freq          : np.ndarray = None,   # [Hz] Frequency array shape(nfreqs,)
                               force         : np.ndarray = None):  # [N] Force array shape(nfreqs, Nnodes, 3)
        """
        Calculate the acoustic pressure at an observer location due to dipole
        sources and their images using the method of images.

        This function generates an image system for dipole sources bounded by
        two horizontal planes, then computes the total pressure field by
        summing contributions from all real and image sources.

        Parameters
        ----------
        nodes_pos_real : ndarray, shape (Nnodes, 3)
            Coordinates of the real source nodes (x, y, z).

        observer_pos : ndarray, shape (3,)
            Coordinates of the observer point (x, y, z).

        freq : ndarray, shape (Nfreqs,)
            Frequencies at which to compute the pressure field [Hz].

        force : ndarray, shape (Nfreqs, 3, Nnodes)
            Frequency-domain force vectors (Fx, Fy, Fz) applied at each real node.

        Upper_BC, Lower_BC : int
            Boundary condition type at the upper and lower planes:
            +1 : Neumann-type (normal derivative of pressure = 0)
            -1 : Dirichlet-type (pressure = 0)

        Upper_HBC, Lower_HBC : float
            z-coordinates of the upper and lower boundary planes.

        N : int, optional
            Number of image layers. Each real node generates 2*N image nodes.
            Default is 1.

        c0 : float, optional
            Speed of sound in the medium [m/s]. Default is 1500 m/s.

        eps : float, optional
            Small regularization parameter to avoid division by zero when
            computing distances. Default is 1e-12.

        Returns
        -------
        p_total : ndarray, shape (Nfreqs,)
            Complex pressure at the observer location for each frequency.
            The real part represents the acoustic pressure in the time domain
            for harmonic sources.
        """
        
        eps = self.eps
        c0 = self.c_wat


        if force.shape[2] == 3: force = force.transpose(0, 2, 1)
        
        # Real z-coordinates
        zi = nodes_pos_real[:, 2]

        # Generate all nodes (real + images)
        z_nodes_all, Force_all, is_real, parent, BC_all = self.method_of_images(zi, force)

        N_nodes_total = z_nodes_all.size
        N_freqs = freq.size

        # Node positions: x,y from parent real node, z from image
        nodes_pos = np.empty((3, N_nodes_total))
        nodes_pos[0, :] = nodes_pos_real[parent, 0]
        nodes_pos[1, :] = nodes_pos_real[parent, 1]
        nodes_pos[2, :] = z_nodes_all

        # Observer position relative to each node
        r_vec = observer_pos[:, None] - nodes_pos
        r_norm = np.linalg.norm(r_vec, axis=0)
        r_norm = np.maximum(r_norm, eps)
        r_hat = r_vec / r_norm[None, :]

        # Wavenumber
        k = 2 * np.pi * freq / c0

        # Reshape arrays for broadcasting
        r_norm_3d = r_norm[None, None, :]   # (1, 1, N_nodes_total)
        k_3d = k[:, None, None]             # (N_freqs, 1, 1)

        dot_product = np.einsum('in,fin->fn', r_hat, Force_all)[:,None,:]

        # Dipole pressure kernel
        amplitude = 1 / (4 * np.pi * r_norm_3d) * (-1j * k_3d + 1 / r_norm_3d)
        phase = np.exp(1j * k_3d * r_norm_3d)

        # Combine all terms and sum over nodes
        p_individual = amplitude * dot_product * phase * BC_all
        p_total = np.sum(p_individual, axis=(1, 2))

        return p_total



