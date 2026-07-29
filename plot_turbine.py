from plotting.plot_style import AcademicStyle
from plotting.plot_utils import parse_args, load_case
import plotting.plot_single_case as p    
import matplotlib.pyplot as plt

def main():
    # 1. Visual asthetics
    AcademicStyle.apply()

    case = load_case(parse_args(), verbose=False)

    # p.plot_structure(case, mode="3D")
    # p.plot_spectrum(case, do_thresholds=False)
    # p.plot_polar(case)
    # p.plot_cylinder(case, mode='SPL')
    p.plot_distance_decay(case, filter_under=1.0)
    plt.show()
    


if __name__ == "__main__":
    main()