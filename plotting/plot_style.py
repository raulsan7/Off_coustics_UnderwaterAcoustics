"""
Module: plot_style.py
Description: -

Author: Raul Sanz Ramirez (raul.sanz.ramirez@upm.es / raul.sanz.ramirez@gmail.com)
Institution: Universidad Politecnica de Madrid - ETSIAE
Date: 07/2026 
"""

import matplotlib.pyplot as plt


class AcademicStyle:
    """
    Configures visual asthetics for scientific publications.
    """

    # Fonts
    FONT_FAMILY   = "serif"
    DEFAULT_CMAP  = "inferno"
    PRESSURE_CMAP = "seismic"
    CASE_COLORS   = ["darkblue", "darkorange", "teal"]    # Monopile, Floating (shallow), Floating (deep)
    BAND_COLORS   = ["steelblue", "tomato", "seagreen", "mediumpurple", "goldenrod", "darkcyan", "salmon", "olivedrab", "slategray", "peru"]
    LINE_STYLES   = ['-', '--', ':', '-.', '-o', '-v', '--o', ':^']


    P_REF = 1e-6            # [Pa] Reference Pressure for water

    @classmethod
    def apply(cls):
        """
        Applies global configuration to Matplotlib
        """
        plt.rcParams.update({
            "font.family": cls.FONT_FAMILY,
            "text.usetex": True,
            "font.size": 11,
            "axes.labelsize": 16,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize":14,
            "figure.titlesize": 20,
            "savefig.dpi": 300,
            "savefig.bbox": 'tight'
        })

