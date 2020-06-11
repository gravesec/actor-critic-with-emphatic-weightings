# ./compute_canada/local_pw_ace_sweep.sh PuddleWorld-v0 5000 1000

#!/bin/bash

#Run this script from the $SCRATCH directory on Niagara to save the output ($HOME is read-only):
#$ sbatch --account=def-whitem --mail-user=imani@ualberta.ca --mail-type=ALL $SCRATCH/actor-critic-with-emphatic-weightings/compute_canada/local_pw_ace_sweep.sh PuddleWorld-v0 5000000 1000000

#SBATCH --job-name=pw_ace_sweep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80  # Change to 80 to use hyperthreading.
#SBATCH --time=00-01:00:00  # DD-HH:MM:SS

# Configure virtual environment locally:
#virtualenv $SLURM_TMPDIR/ve
#source $SLURM_TMPDIR/ve/bin/activate
#pip install --upgrade pip
#pip install -e $SCRATCH/gym-puddle
#pip install -r $SCRATCH/actor-critic-with-emphatic-weightings/requirements.txt

date

source ve/bin/activate

module load python/3.6.5

echo $1
echo $2
echo $3

python generate_experience.py \
--experiment_name local_res/$1 \
--num_runs 3 \
--num_timesteps $2 \
--random_seed 3139378768 \
--num_cpus -1 \
--backend "loky" \
--verbosity 0 \
--behaviour_policy "lambda s: np.ones(env.action_space.n)/env.action_space.n" \
--environment $1

python run_low_var_ace.py \
--experiment_name local_res/$1 \
--checkpoint_interval $3 \
--num_cpus 1 \
--backend "loky" \
--verbosity 0 \
--interest_function "lambda s, g=1: 1. if g==0. else 0." \
--behaviour_policy "lambda s: np.ones(env.action_space.n)/env.action_space.n" \
--environment $1 \
--run_mode "combinations" \
--gamma 1.0 \
--alpha_a .1 \
--alpha_c .01 \
--alpha_c2 .001 \
--lambda_c .1 \
--eta 1. \
--num_tiles_a 2 2 \
--num_tilings_a 2 \
--num_tiles_c 3 3 \
--num_tilings_c 3 \
--bias_unit 1

python evaluate_policies.py \
--experiment_name local_res/$1 \
--num_evaluation_runs 5 \
--max_timesteps 5000 \
--random_seed 1944801619 \
--num_cpus -1 \
--backend "loky" \
--verbosity 0 \
--objective "episodic" \
--environment $1

date