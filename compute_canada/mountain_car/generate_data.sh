#!/bin/bash

## Note: change account to def-whitem if running as part of Martha's allocation:
#SBATCH --account=def-sutton
#SBATCH --job-name=generate_data
#SBATCH --mail-user=graves@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --time=00:15:00

parallel --jobs $SLURM_NTASKS --joblog logs/generate_data.log --resume python scripts/mountain_car/generate_data.py ::: {0..49} ::: 50000
