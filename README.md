## Installation:

1. Install python3 (3.7.5) if necessary.
On MacOS using [Homebrew](https://brew.sh) to install python3 is pretty good.

2. Create a new python virtual environment in the 'actor-critic-with-emphatic-weightings' directory (named 've' in this case):
```
$ python3 -m venv ve
```

3. Activate the virtual environment:
```
$ source ve/bin/activate
```

4. Install the required python package dependencies:
```
(ve)$ pip install -r requirements.txt
```

## Running the experiment scripts on Compute Canada:
1. Activate the virtual environment (if necessary):
```
$ cd $SCRATCH/actor-critic-with-emphatic-weightings/
$ source ve/bin/activate
```

2. Read the help output for each script to determine which arguments you want to run your experiment with.
```
(ve)$ python generate_experience.py --help
(ve)$ python sweep.py --help
(ve)$ python run_ace.py --help
(ve)$ python evaluate_policies.py --help
```

2. Generate the data to use to train the agents:
```
(ve)$ python generate_experience.py
```
Generating the data ahead of time is more efficient than doing it for each agent, and is possible due to off-policy learning.

3. Run the sweep.py python script to generate bash scripts for SLURM to run:
```
(ve)$ python sweep.py
```
The script will give you a really rough estimate of how long the job might take and the number of nodes necessary to complete the job in the amount of time specified via the "--num_hours" argument.
If requesting that number of nodes is ok with you, type "y", hit enter, and the script will generate the individual bash scripts for each node.

4. Schedule the generated script(s) to run via SLURM:
```
(ve)$ sbatch mountain-car/sweep0.sh
```

5. Evaluate the resulting policies:
```
(ve)$ python evaluate_policies.py
```

6. Use a jupyter notebook to explore the data and generate plots of performance.