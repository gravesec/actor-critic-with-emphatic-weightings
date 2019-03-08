#!/bin/bash

## Note: change account to def-whitem if running as part of Martha's allocation:
#SBATCH --account=def-sutton
#SBATCH --job-name=returns_eac
#SBATCH --mail-user=graves@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00-2:00

use_excursions=(0)
bias_unit=(0 1)
alpha_u=(.0001 .001 .01 .1 1)
alpha_v=(.0001 .001 .01 .1 1)
alpha_w=(.0001 .001 .01 .1 1)
lamda_v=($2)
lamda_a=($1)
run=({0..49})
num_eval_runs=(5)
max_timesteps=(1000)

parallel --jobs $SLURM_NTASKS --joblog logs/compute_returns_eac$1-$2.log --resume python scripts/mountain_car/compute_returns_eac.py ::: ${use_excursions[@]} ::: ${bias_unit[@]} ::: ${alpha_u[@]} ::: ${alpha_v[@]} ::: ${alpha_w[@]} ::: ${lamda_v[@]} ::: ${lamda_a[@]} ::: ${run[@]} ::: ${num_eval_runs[@]} ::: ${max_timesteps[@]}
