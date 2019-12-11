## Installation:

1. Install python3 (3.7.5) if necessary.
I'm mostly using PyCharm these days.
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

## Running the mountain car experiment scripts:
1. Activate the virtual environment (if necessary):
```
$ source ve/bin/activate
```

2. Change to the mountain_car directory and run scripts:
```
(ve)$ cd mountain_car
(ve)$ python generate_experience.py
(ve)$ python run_ace.py
(ve)$ python evaluate_policies.py
(ve)$ python plot_performance.py
```

3. Deactivate the virtual environment when finished:
```
(ve)$ deactivate
```

## About the scripts:

### generate_experience.py
The **generate_experience.py** script runs the given behaviour policy on [OpenAI Gym's](https://github.com/openai/gym) implementation of the [mountain car environment](https://en.wikipedia.org/wiki/Mountain_car_problem) for the given number of independent runs each containing the given number of total timesteps of experience [in parallel](https://joblib.readthedocs.io/en/latest/), and saves the results to a [memmapped](https://joblib.readthedocs.io/en/latest/auto_examples/parallel_memmap.html) [numpy structured array](https://docs.scipy.org/doc/numpy/user/basics.rec.html) in the given subdirectory.

We're using a structured array so that information about each field is saved with the data, and we're memmapping it so that multiple processes can write to the structured array in parallel (otherwise we exceed Compute Canada's limit on number of files).

The behaviour policy is specified as a string containing a lambda function that gets eval'd to return the policy function. Given the current state, the policy function returns a vector containing the probability of taking each action. Two examples are available in the help by running:
```
(ve)$ python generate_experience.py -h
```
 
 The command line arguments are also saved to a file called 'generate_experience.args' in the given subdirectory. This file format can be used to pass command line arguments very easily:
 ```
(ve)$ python generate_experience.py @experiment/generate_experience.args
```

### run_ace.py
The **run_ace.py** script runs the ACE algorithm with the given parameter settings, interest function, and behaviour policy in parallel, and saves the learned policy after every N timesteps of experience.

A good way to get familiar with the scripts is to run them with their default parameters and inspect the help menus and the generated .args files.