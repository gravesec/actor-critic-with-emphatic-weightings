#!/bin/bash

## Note: change account to def-whitem if running as part of Martha's allocation:
#SBATCH --account=def-sutton
#SBATCH --job-name=run_eac
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=512
#SBATCH --cpus-per-task=15

# actor step-size
alpha_u=(0.01 0.02 0.05 0.1 0.2 0.5 1.0)

lamda_a=(0.0 0.25 0.5 0.75 1.0)

# These parameters are not used in this experiment.
true_m=(0)
bias_unit=(0)

# These parameters are not used in this experiment.
alpha_v=(.005)
alpha_w=(.00001)
lamda=(1.0)

gamma=(1.0)
run=({0..29})
num_eval_points=(50)

parallel --jobs 1 --progress python scripts/tiny_counterexample/run_eac_oracle.py ::: ${bias_unit[@]} ::: ${true_m[@]} ::: ${alpha_u[@]} ::: ${alpha_v[@]} ::: ${alpha_w[@]} ::: ${lamda[@]} ::: ${lamda_a[@]} ::: ${gamma[@]} ::: ${run[@]} ::: ${num_eval_points[@]}
