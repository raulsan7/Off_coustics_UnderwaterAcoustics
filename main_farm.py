"""
Module: main_farm.py
Description: Main script to run a WindFarm acoustic simulation.

Author: Raul Sanz Ramirez (raul.sanz.ramirez@upm.es / raul.sanz.ramirez@gmail.com)
Institution: Universidad Politecnica de Madrid - ETSIAE
Date: 08/2026 
"""

import time
import numpy as np
from core.WindFarm import WindFarm
from core.AcousticMethods import MethodImages
from core.TurbineTypes import DTU10MWFloating, DTU10MWMonopile
import tracemalloc
import resource
import traceback

def main() -> None:

    # Define gloabl parameters
    GLOBAL_DEPTH = 30.0

    # Initialize turbines
    t1 = DTU10MWMonopile(
        rootname   = "DTU_DeltaWind_mn_ws11.4",
        output_dir = "./OP_output/",
        WindSpeed  = 11.4,
        Depth      = GLOBAL_DEPTH,
        WindDir    = 0.0,
        AxisPos    = [0.0, 0.0],
        Nmembers   = 8,
        Nnodes     = 5,)

    t2 = DTU10MWMonopile(
        rootname   = "DTU_DeltaWind_mn_ws11.4",
        output_dir = "./OP_output/",
        WindSpeed  = 11.4,
        Depth      = GLOBAL_DEPTH,
        WindDir    = 45.0,
        AxisPos    = [200.0, 0.0],
        Nmembers   = 8,
        Nnodes     = 5,)


    # Define WindFarm
    turbines = [t1, t2]
    farm = WindFarm(turbines=turbines, debug=True)
    farm.read_input(verbose=True)
    farm.compute_force(verbose=True, filter_freqs=True)
    acoustic_model = MethodImages(system=farm, N_images=30, Upper_HBC=0., Lower_HBC=-farm.Depth)
    farm.set_acoustic_solver(acoustic_model)

    # Run pressure computations
    farm.run_spectrums()
    # farm.run_polar()
    # farm.run_line()
    # farm.run_cylinder()
    # farm.run_sliceXY()
    # farm.run_sliceXZ()
    # farm.run_sliceVertical()
    # farm.run_spheres()
    # farm.run_all()



if __name__ == "__main__":
    start = time.time()
    tracemalloc.start()
    try:
        main()
    except Exception:
        # Print traceback so user sees the error
        traceback.print_exc()
        raise
    finally:
        end = time.time()
        elapsed_time = end - start

        if elapsed_time > 60. and elapsed_time <= 3600.:
            elapsed_time /= 60; tag = "min"
        elif elapsed_time > 3600 and elapsed_time <= 86400.:
            elapsed_time /= 3600.; tag = "h"
        elif elapsed_time > 86400.:
            elapsed_time /= 86400.; tag = "days"
        else:
            tag = "s"

        # Process-level peak RSS (platform-dependent units; on Linux it's KB)
        try:
            ru = resource.getrusage(resource.RUSAGE_SELF)
            peak_rss_kb = getattr(ru, 'ru_maxrss', None)
            peak_rss_mb = peak_rss_kb / 1024.0 if peak_rss_kb is not None else None
        except Exception:
            peak_rss_mb = None

        # tracemalloc peak (Python allocations)
        try:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            tracemalloc_peak_mb = peak / (1024.0 ** 2)
        except Exception:
            tracemalloc_peak_mb = None

        print(f"\n\n    Total elapsed time {elapsed_time:.3f} ({tag})")
        if peak_rss_mb is not None:
            print(f"    Peak RSS (getrusage) : {peak_rss_mb:.2f} MB")
        else:
            print("    Peak RSS (getrusage) : N/A")

        if tracemalloc_peak_mb is not None:
            print(f"    Peak tracemalloc    : {tracemalloc_peak_mb:.2f} MB")
        else:
            print("    Peak tracemalloc    : N/A")