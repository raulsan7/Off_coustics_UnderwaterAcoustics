import numpy as np
import matplotlib.pyplot as plt
import plotting.plot_single_case as p
import plotting.plot_comparison as pc
from plotting.plot_style import AcademicStyle
from plotting.plot_utils import parse_args, load_cases

def main():
    # 1. Visual asthetics
    AcademicStyle.apply()

    cases = load_cases(parse_args(), verbose=False)

    if len(cases) != 3:
        # p.plot_structure(cases, mode="3D")
        # p.plot_spectrum(cases, do_thresholds=False)
        # p.plot_polar(cases)
        # p.plot_cylinder(cases, filter_under=1.0)
        # p.plot_distance_decay(cases, filter_under=1.0, turbine_data=True)
        p.plot_line(cases)
        # p.plot_sliceXY(cases, filter_under=1.0)
        # p.plot_sliceXZ(cases, filter_under=1.0)
        # p.compute_sphere_metrics(cases, filter_under=1.0)
        # p.compute_cylinder_metrics(cases, filter_under=1.0)
    else:
        # pc.plot_spectrum(cases)
        # pc.plot_polar(cases)
        # pc.plot_cylinder(cases, filter_under=1.0)
        # pc.plot_distance_decay(cases, filter_under=1.0, turbine_data=True)
        # pc.plot_line(cases)
        # pc.plot_sliceXY(cases, filter_under=1.0)
        pc.plot_sliceXZ(cases, filter_under=1.0)
        # pc.compute_sphere_metrics(cases, filter_under=1.0)
        # pc.compute_cylinder_metrics(cases, filter_under=1.0)

    plt.show()


if __name__ == "__main__":
    main()