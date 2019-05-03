import os
import numpy as np


def save_data_to_file(data, file_path):
    """
    Saves a list of numpy arrays to a compressed .npz file.
    Args:
        data: a list of numpy arrays of variable lengths.
        file_path: the path of the desired output file (e.g. 'output/experiment/data').
    """
    # Transform the list of arrays into a 1D array and an array of indexes:
    lengths = [array.shape[0] for array in data]
    indices = np.cumsum(lengths[:-1])
    stacked = np.concatenate(data, axis=0)

    # Save the 1D array and index array:
    create_directory(file_path)
    np.savez_compressed(file_path, stacked_array=stacked, index_array=indices)


def load_data_from_file(file_path):
    """
    Loads a compressed .npz file and returns a list of numpy arrays.
    Args:
        file_path: the path to the data file to load.
    Returns:
        A python list of numpy arrays of variable lengths.
    """
    npzfile = np.load(file_path)
    indices = npzfile['index_array']
    stacked = npzfile['stacked_array']
    return np.split(stacked, indices, axis=0)


def save_policy_to_file(policy, file_path):
    """
    Saves a discrete policy to a compressed file.
    Args:
        policy: a numpy array of action feature preferences.
        file_path: the path of the desired output file.
    """
    create_directory(file_path)
    np.savez_compressed(file_path, policy=policy)


def load_policy_from_file(file_path):
    """
    Loads a discrete policy from a file.
    Args:
        file_path: the path to load the policy from.
    Returns:
        A numpy array of action feature preferences.
    """
    npzfile = np.load(file_path)
    return npzfile['policy']


def save_dictionary_to_file(dictionary, file_path):
    """
    Saves a dictionary mapping strings to arrays to a file.
    """
    create_directory(file_path)
    np.savez_compressed(file_path, **dictionary)


def load_dictionary_from_file(file_path):
    npzfile = np.load(file_path)
    return npzfile


def save_returns_to_file(returns, file_path):
    """
    Saves an array of returns to a file.
    """
    create_directory(file_path)
    # np.savez_compressed(file_path, returns=returns)
    np.savetxt(file_path, returns)


def load_returns_from_file(file_path):
    """
    Loads an array of returns from a file.
    """
    npzfile = np.loadtxt(file_path)
    return npzfile#['returns']


def create_directory(file_path):
    # Create the directory if it doesn't already exist:
    directory_name = os.path.dirname(file_path)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name, exist_ok=True)


def save_args_to_file(args, args_file_path):
    """
    Saves command line arguments to a file in a format interpretable by argparse (one 'word' per line).
    :param args: Namespace of command line arguments.
    :param args_file_path: Path to the file to save the arguments in.
    :return:
    """
    with open(args_file_path, 'w') as args_file:
        for key, value in vars(args).items():
            if isinstance(value, list):
                value = '\n'.join(str(i) for i in value)
            args_file.write('--{}\n{}\n'.format(key, value))
