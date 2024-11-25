from core import ContinuousOptimization
from lshade import LSHADE
from correction_handler import *
import os
from pathlib import Path

# Define correction methods
correction_methods = [
    METHOD_VECTOR_BEST, METHOD_VECTOR_TARGET, METHOD_SATURATION,
    METHOD_MIDPOINT_TARGET, METHOD_MIDPOINT_BEST,
    METHOD_EXPC_TARGET, METHOD_EXPC_BEST, METHOD_UNIF, METHOD_MIRROR, METHOD_TOROIDAL]

desktop_path = Path.home() / "Desktop"
results_dir = desktop_path / "evo_dataset_15nov"
results_dir.mkdir(exist_ok=True)



# Define problem dimensions
problem_dimensions = {
    # "HNO": 5,
    # "TCSD": 3,
    # "IBD": 4,
    # "CBD": 5,
    # "TBTD": 2,
    "CBHD": 4,
    "TCD": 2,
    "PVD":4,
    "SRD":7

}
# problem_dimensions = {
#     "IBD": 4,
#
# }
num_runs = 10

strategies = ["DEB","PENALTY"]
pbs = list(problem_dimensions.keys())

for pb in pbs:
    d = problem_dimensions[pb]
    print(f"Problem: {pb}, Dimension: {d}")
    problem = ContinuousOptimization(dim=d, fct_name=pb, inst=1)

    for corr in correction_methods:
        for strat in strategies:
            for run in range(num_runs):
                lshade = LSHADE(
                    problem=problem,
                    pop_size=100,
                    sizeH=6,
                    NFE_max=30000,
                    N_min=10,
                    corr_type=None,
                    corr_method=corr,
                    run_number=run + 21,
                    path=results_dir,
                    constraint_strategy=strat
                )
                best_solution = lshade.optimize()
