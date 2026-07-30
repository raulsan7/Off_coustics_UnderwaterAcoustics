"""
Module: main_turbine.py
Description: Main execution entry point for individual offshore wind turbine 
             vibroacoustic analysis. 

             Execution Pipeline:
             ┌────────────────────────────────────────────────────────┐
             │                  Initialize Turbine:                   │
             │       Setup geometry, wind speed, & paths              │
             └──────────────────────────┬─────────────────────────────┘
                                        ▼
             ┌────────────────────────────────────────────────────────┐
             │                     read_input():                      │
             │   Parse SubDyn nodes and downsample accelerations     │
             └──────────────────────────┬─────────────────────────────┘
                                        ▼
             ┌────────────────────────────────────────────────────────┐
             │                   compute_force():                     │
             │ Calculate frequency-domain dipole forces and filter    │
             └────────────────────────────────────────────────────────┘

Author: Raul Sanz Ramirez (raul.sanz.ramirez@upm.es / raul.sanz.ramirez@gmail.com)
Institution: Universidad Politecnica de Madrid - ETSIAE
Date: 07/2026 
"""

import time
from core.AcousticMethods import MethodImages
from core.TurbineTypes import DTU10MWFloating, DTU10MWMonopile, SAITEC2MWFloating

def main() -> None:

    t = DTU10MWMonopile(
        rootname   = "DTU_DeltaWind_mn_ws11.4",
        save_name  = "plot_mn_SD30",
        WindSpeed  = 11.4,
        Depth      = 30.0,
        Nmembers   = 8,
        Nnodes     = 5,
        save_dir   = "test"
    )

    # t = DTU10MWFloating(
    #     rootname   = "DTU_DeltaWind_fl_ws11.4",
    #     save_name  = "plot_fl_SD30",
    #     WindSpeed  = 11.4,
    #     Depth      = 350.0,
    #     Nmembers   = 9,
    #     Nnodes     = 9,
    # )

    # t = DTU10MWFloating(
    #     rootname   = "DTU_DeltaWind_fl_ws11.4",
    #     save_name  = "plot_fl_SD30_shallow",
    #     WindSpeed  = 11.4,
    #     Depth      = 30.0,
    #     Nmembers   = 9,
    #     Nnodes     = 9,
    # )

    # t = SAITEC2MWFloating(
    #     rootname = "SENVION_2MW_fl_ws15.0",
    #     save_name = "SAITEC",
    #     WindSpeed = 15.,
    #     Depth = 80.,
    #     Nmembers = 7,
    #     Nnodes = 9
    # )

    t.read_input(verbose=True)
    t.compute_force(verbose=True, filter_freqs=True)

    acoustic_model = MethodImages(system=t, N_images=30, Upper_HBC=0., Lower_HBC=-t.Depth)
    t.set_acoustic_method(acoustic_model)
    # t.run_spectrums()
    # t.run_polar()
    # t.run_cylinder()
    # t.run_decay()
    # t.run_line()
    # t.run_sliceXY()
    # t.run_sliceXZ()
    # t.run_sphere()
    t.run_all()

    

if __name__ == "__main__":

    start = time.time()
    main()
    end = time.time()

    elapsed_time = end-start

    if elapsed_time > 60. and elapsed_time <= 3600.:
        elapsed_time /= 60; tag = "min"
    elif elapsed_time > 3600 and elapsed_time <= 86400.:
        elapsed_time /= 3600.; tag = "h"
    elif elapsed_time > 86400.:
        elapsed_time /= 86400.; tag = "days"
    else:
        tag = "s"

    print(f"\n\n    Total elapsed time {elapsed_time} ({tag})")
