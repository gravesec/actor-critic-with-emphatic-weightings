#!/bin/bash

## Note: change account to def-whitem if running as part of Martha's allocation:
#SBATCH --account=def-sutton
#SBATCH --job-name=run_eac
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=512
#SBATCH --cpus-per-task=15

#alpha_u=(0.01 0.02 0.05 0.1 0.2 0.5 1.0)
alpha_u=(0.00005 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01)
lamda_a=(0.0 0.25 0.5 0.75 1.0)
#lamda_a=(0.0)

true_m=(0)
bias_unit=(0)

alpha_v=(.005)
alpha_w=(.00001)
lamda=(1.0)

gamma=(1.0)
run=({0..9})
num_eval_points=(20)

parallel --jobs 8 --progress python scripts/long_counterexample/run_eac_oracle.py ::: ${bias_unit[@]} ::: ${true_m[@]} ::: ${alpha_u[@]} ::: ${alpha_v[@]} ::: ${alpha_w[@]} ::: ${lamda[@]} ::: ${lamda_a[@]} ::: ${gamma[@]} ::: ${run[@]} ::: ${num_eval_points[@]}
