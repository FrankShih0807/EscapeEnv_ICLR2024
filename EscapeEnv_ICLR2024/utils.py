import os
import numpy as np
import argparse
from ruamel.yaml import YAML
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from EscapeEnv_ICLR2024.common.base_agent import BaseAgent
from EscapeEnv_ICLR2024 import DQN, LKTD_SARSA, SGHMC_SARSA, BootDQN, KOVA, QRDQN
yaml = YAML()
yaml.preserve_quotes = True


ALGOS: Dict[str, Type[BaseAgent]] = {
    "dqn": DQN,
    "lktd_sarsa": LKTD_SARSA,
    "sghmc_sarsa": SGHMC_SARSA,
    "boot_dqn": BootDQN,
    "kova": KOVA,
    "qrdqn": QRDQN,
}

def create_log_folder(path):
    os.makedirs(path, exist_ok=True)

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.load(file)

def save_yaml(config, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(config, file)
        
def create_parser():
    parser = argparse.ArgumentParser(description='Initial Argument Parser')
    parser.add_argument('--algo', help="RL Algorithm", type=str, default="dqn", required=False, choices=list(ALGOS.keys()))
    parser.add_argument('--task_id', type=int, default=-1)

    parser.add_argument('--exp_name', type=str, default=None)
    
    parser.add_argument(
        "-params",
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",
    )
    return vars(parser.parse_args())

def create_output_dir(inital_args):
    default_output_path = os.path.join(Path(__file__).parent.parent, 'output')
    if inital_args['exp_name'] is None:
        inital_args['exp_name'] = '{}-test'.format(inital_args['algo'])
    path = os.path.join(default_output_path, inital_args['exp_name'])
    os.makedirs(path, exist_ok=True)
    
    if inital_args['task_id'] >= 0:
        exp_name = 'exp_{}'.format(inital_args['task_id'])
    else:
        task_id = 0
        while 'exp_{}'.format(task_id) in os.listdir(path):
            task_id += 1
        exp_name = 'exp_{}'.format(task_id)       
    exp_path = os.path.join(path, exp_name)    
    os.makedirs(exp_path, exist_ok=True)
    return exp_path

def update_hyperparams(original_params, new_params):
    if new_params is not None:
        for key, value in new_params.items():
            
            if find_key_in_dict(original_params, key, value):
                print('update {}: {}'.format(key, value))
            else:
                raise KeyError(f"Hyperparameter '{key}' not found.")
            
class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.
    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)    
        
        
def find_key_in_dict(d, key_to_find, new_value):
    """
    Check if key_to_find is in the dictionary d. This function
    searches recursively in all nested dictionaries.
    
    :param d: Dictionary in which to search for the key.
    :param key_to_find: Key to search for.
    :return: True if the key is found, False otherwise.
    """
    if key_to_find in d:
        d[key_to_find] = new_value
        return True
    for key, value in d.items():
        if isinstance(value, dict):
            if find_key_in_dict(value, key_to_find, new_value):
                return True
    return False



if __name__ == '__main__':
    nested_dict = {
        'a': 1,
        'b': {'c': 2, 'd': {'e': 3}},
        'f': 4
    }
