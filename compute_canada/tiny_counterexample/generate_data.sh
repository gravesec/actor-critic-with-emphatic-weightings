#!/bin/bash

## Note: change account to def-whitem if running as part of Martha's allocation:
#SBATCH --account=def-whitem
#SBATCH --job-name=generate_data
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=512
#SBATCH --cpus-per-task=15

# Use GNU parallel to run the generate_data.py python script 30 times, passing in the run number and number of episodes:
# parallel --jobs 15 --progress --shuf python generate_data.py ::: {0..29} ::: 100
parallel --jobs 8 --progress python scripts/tiny_counterexample/generate_data.py ::: {0..29} ::: 2000
