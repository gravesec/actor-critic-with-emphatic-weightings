#!/bin/bash

#SBATCH --account=def-sutton
#SBATCH --job-name=ace_mc_sweep
#SBATCH --mail-user=graves@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00-01:00:00     # DD-HH:MM:SS

# Prepare virtual environment on compute node:
virtualenv --no-download $SLURM_TMPDIR/ve
source $SLURM_TMPDIR/ve/bin/activate
pip install --no-index -r ~/actor-critic-with-emphatic-weightings/requirements.txt

# Generate data on compute node:
python ~/actor-critic-with-emphatic-weightings/generate_experience.py \
--experiment_name $SLURM_TMPDIR/ace_mc_sweep \
--num_runs 30 \
--num_timesteps 200000 \
--random_seed 3139378768 \
--num_cpus -1 \
--backend "loky" \
--verbosity 0 \
--behaviour_policy "lambda s: np.ones(env.action_space.n)/env.action_space.n" \
--environment "MountainCar-v0"

# Run ace parameter sweep, saving to compute node:
python ~/actor-critic-with-emphatic-weightings/run_ace.py \
--experiment_name $SLURM_TMPDIR/ace_mc_sweep \
--checkpoint_interval 5000 \
--num_cpus -1 \
--backend "loky" \
--verbosity 0 \
--interest_function "lambda s, g=1: 1." \
--behaviour_policy "lambda s: np.ones(env.action_space.n)/env.action_space.n" \
--environment "MountainCar-v0" \
--run_mode "combinations" \
--gamma 1.0 \
--alpha_a 0.01 \
--alpha_c 0.05 \
--alpha_c2 0.0001 \
--lambda_c 0.0 \
--eta 0.0 \
--num_tiles 9 \
--num_tilings 9 \
--num_features 100000 \
--bias_unit 1

python ~/actor-critic-with-emphatic-weightings/evaluate_policies.py \
--experiment_name $SLURM_TMPDIR/ace_mc_sweep \
--num_evaluation_runs 5 \
--max_timesteps 1000 \
--random_seed 1944801619 \
--num_cpus -1 \
--backend "loky" \
--verbosity 0 \
--objective "episodic" \
--environment "MountainCar-v0"

# Copy the results to the home directory:
tar -cf ~/actor-critic-with-emphatic-weightings/ace_mc_sweep.tar $SLURM_TMPDIR/ace_mc_sweep