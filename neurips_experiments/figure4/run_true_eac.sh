#!/bin/bash

## Note: change account to def-whitem if running as part of Martha's allocation:
#SBATCH --account=def-sutton
#SBATCH --job-name=run_eac
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=512
#SBATCH --cpus-per-task=15

# Actor step-size
alpha_u=(0.00005 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01)

lamda_a=(1.0)

# Whether the agent uses the true emphatic weightings. When true_m is 1, lambda_a is ignored.
true_m=(1)

bias_unit=(0)

# These parameters are not used in this experiment
alpha_v=(.005)
alpha_w=(.00001)
lamda=(1.0)

gamma=(1.0)
run=({0..9})
num_eval_points=(20)

parallel --jobs 1 --progress python scripts/long_counterexample/run_eac_oracle.py ::: ${bias_unit[@]} ::: ${true_m[@]} ::: ${alpha_u[@]} ::: ${alpha_v[@]} ::: ${alpha_w[@]} ::: ${lamda[@]} ::: ${lamda_a[@]} ::: ${gamma[@]} ::: ${run[@]} ::: ${num_eval_points[@]}
