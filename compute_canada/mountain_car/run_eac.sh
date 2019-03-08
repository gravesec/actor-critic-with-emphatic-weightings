#!/bin/bash

## Note: change account to def-whitem if running as part of Martha's allocation:
#SBATCH --account=def-sutton
#SBATCH --job-name=eac
#SBATCH --mail-user=graves@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00-1:00

bias_unit=(0)
alpha_u=(.0001 .001 .01 .1 1)
alpha_v=(.0001 .001 .01 .1 1)
alpha_w=(.0001 .001 .01 .1 1)
lamda_v=($2)
lamda_a=($1)
run=({0..49})
checkpoint_interval=(1000)

parallel --jobs $SLURM_NTASKS --joblog logs/eac$1-$2.log --resume python scripts/mountain_car/run_eac.py ::: ${bias_unit[@]} ::: ${alpha_u[@]} ::: ${alpha_v[@]} ::: ${alpha_w[@]} ::: ${lamda_v[@]} ::: ${lamda_a[@]} ::: ${run[@]} ::: ${checkpoint_interval[@]}
