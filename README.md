# Installation:

1. Install python3, if necessary. On MacOS, using Homebrew to install python3 is pretty good.

2. Create a new python virtual environment in the 'off-policy-actor-critic' directory (named 've' in this case):
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

# Running the experiment scripts:
1. Activate the virtual environment (if necessary):
```
$ source ve/bin/activate
```

2. Run the desired scripts from the 'off-policy-actor-critic' directory:
```
(ve)$ python scripts/tiny_counterexample/generate_data.py
(ve)$ python scripts/tiny_counterexample/run_offpac.py
(ve)$ python scripts/tiny_counterexample/evaluate_policies.py
```

3. Deactivate the virtual environment when finished:
```
(ve)$ deactivate
```

# Running the unit tests:
```
(ve)$ python -m unittest discover
```
