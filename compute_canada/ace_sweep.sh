#!/bin/bash

#Run this script from the $SCRATCH directory on Niagara to save the output ($HOME is read-only):
#$ sbatch --account=def-sutton --mail-user=graves@ualberta.ca --mail-type=ALL $SCRATCH/actor-critic-with-emphatic-weightings/compute_canada/ace_sweep.sh MountainCar-v0 100000 10000

#SBATCH --job-name=ace_sweep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80  # Change to 80 to use hyperthreading.
#SBATCH --time=00-01:00:00  # DD-HH:MM:SS

module load python/3.6.5

# Configure virtual environment locally:
#virtualenv $SLURM_TMPDIR/ve
#source $SLURM_TMPDIR/ve/bin/activate
#pip install --upgrade pip
#pip install -e $SCRATCH/gym-puddle
#pip install -r $SCRATCH/actor-critic-with-emphatic-weightings/requirements.txt

source $SCRATCH/actor-critic-with-emphatic-weightings/ve/bin/activate

python $SCRATCH/actor-critic-with-emphatic-weightings/generate_experience.py \
--experiment_name $SCRATCH/$1 \
--num_runs 10 \
--num_timesteps $2 \
--random_seed 3139378768 \
--num_cpus -1 \
--backend "loky" \
--verbosity 0 \
--behaviour_policy "lambda s: np.ones(env.action_space.n)/env.action_space.n" \
--environment $1

python $SCRATCH/actor-critic-with-emphatic-weightings/run_ace.py \
--experiment_name $SCRATCH/$1 \
--checkpoint_interval $3 \
--num_cpus -1 \
--backend "loky" \
--verbosity 0 \
--interest_function "lambda s, g=1: 1." \
--behaviour_policy "lambda s: np.ones(env.action_space.n)/env.action_space.n" \
--environment $1 \
--run_mode "combinations" \
--gamma 1.0 \
--alpha_a .00001 .00005 .0001 .0005 .001 .005 .01 .05 .1 \
--alpha_c .00001 .00005 .0001 .0005 .001 .005 .01 .05 .1 \
--alpha_c2 .0 .00001 .00005 .0001 .0005 .001 .005 .01 .05 .1 \
--lambda_c .0 .4 .7 .9 .99 \
--eta 0. 1. \
--num_tiles 10 10 \
--num_tilings 10 \
--bias_unit 1

python $SCRATCH/actor-critic-with-emphatic-weightings/evaluate_policies.py \
--experiment_name $SCRATCH/$1 \
--num_evaluation_runs 5 \
--max_timesteps 5000 \
--random_seed 1944801619 \
--num_cpus -1 \
--backend "loky" \
--verbosity 0 \
--objective "episodic" \
--environment $1
