import matplotlib.pyplot as plt
from plotting.plot_utils import parse_farm_args, load_farm
from plotting.plot_style import AcademicStyle
import plotting.plot_farm_module as p


def main():

    # 1. Visual asthetics
    AcademicStyle.apply()

    case = load_farm(parse_farm_args(), verbose=False)

    # Select plot
    # p.plot_structure(case, mode='xy')
    # p.plot_spectrums(case, do_thresholds=True)
    # p.plot_polar(case)
    # p.plot_cylinder(case)
    # p.plot_line(case)
    p.plot_sliceXY(case)
    p.plot_sliceXZ(case)
    p.plot_sliceVertical(case)


    plt.show()



if __name__ == "__main__":
    main()