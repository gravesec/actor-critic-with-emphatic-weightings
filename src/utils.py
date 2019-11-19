import os


def save_args_to_file(args, args_file_path):
    """
    Saves command line arguments to a file in a format interpretable by argparse (one 'word' per line).
    :param args: Namespace of command line arguments.
    :param args_file_path: Path to the file to save the arguments in.
    :return:
    """
    os.makedirs(args_file_path.parent, exist_ok=True)
    with open(args_file_path, 'w') as args_file:
        for key, value in vars(args).items():
            if key == 'parameters' and isinstance(value, list):  # Special case for parameters argument.
                for plist in value:
                    args_file.write('--{}\n{}\n'.format(key, '\n'.join(str(i) for i in plist)))
            elif isinstance(value, list):
                value = '\n'.join(str(i) for i in value)
                args_file.write('--{}\n{}\n'.format(key, value))
            else:
                args_file.write('--{}\n{}\n'.format(key, value))