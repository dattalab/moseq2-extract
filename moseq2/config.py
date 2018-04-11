'''
Author: @wingillis
Module to handle operations dealing with moseq configurations.
There are two ways to define moseq parameters:
    - using the cli-based flags
    - using a configuration file
Any cli-based flags will override parameters defined in a config file
'''
from os.path import join
from typing import Dict
import yaml

# an exception class for describing incorrectly formatting config files
class InvalidConfiguration(Exception):
    pass

def create_config() -> Dict:
    '''Generate a new configuration file with all parameters set at their
    default values.
    Returns:
        A dict with default configuration values
    '''
    background = {
        'roi_dilate': (10, 10), # how much to dilate the environment (env) floor to include walls
        'roi_shape': 'ellipse', # shape of the floor dilation (rect for square envs and ellipse for circle envs)
        'roi_index': 0, # if there are multiple envs in one recording, select which roi to use here
        'roi_weights': (1, .1, 1), # (area, extent, distance) Which features matter most for background selection
        'use_plane_bground': False # if the mouse does not move a lot, use a plane instead of the background ROI
    }
    extract = {
        'crop_size': (80, 80),
        'min_height': 10, # minimum height of mouse from floor (mm)
        'max_height': 100,
        'fps': 30,
        'flip_file': None, # filepath for the flip classifier
        'chunk_size': 1000, # 1000 frames per chunk to be processed
        'chunk_overlap': 60, # 60 frames of overlap per chunk
        'em_tracking': False, # extract data with a cable in it (i.e. use em tracking)
        'write_movie': True, # write movie of the extracted mouse results into an mp4
        'prefilter_time': tuple(), # a kernel to filter the mouse temporally
        'prefilter_space': (3, ), # a kernel to filter the mouse spatially
        'output_dir': 'proc' # relative path from the extract file for saving the data
    }
    cables = {
        # in future we will add params for cable extractions
    }
    # return a dict of dicts - this keeps params separated by type
    return {'background': background, 'extract': extract, 'cables': cables}


def load_config(fpath: str) -> Dict:
    '''Loads a configuration file from `fpath` and tests to make sure
    all top level keys are present'''
    with open(fpath, 'r') as f:
        config = yaml.load(f)
    keys = set(flatten_config(config).keys())

    # test to make sure all param types are present
    test_keys = set(flatten_config(create_config()).keys())
    diffs = test_keys - keys
    if len(diffs) > 0:
        raise InvalidConfiguration('Configuration file does not contain necessary keys:\n    {}'
                                   .format('\n    '.join(list(diffs))))
    return config


# NOTE: probably not going to implement this function
def find_config(fpath: str=None) -> str:
    '''Hierarchically search for a config file in 4 places:
        1. The current directory
        2. The parent directory
        3. The home directory
        4. The location where Moseq2 is installed
    Params:
        fpath
    The config file name has to have 'moseq' and 'yaml' in the name.
    Returns:
        the file path where the most important config file is found
    '''
    return


def save_config(fpath: str):
    '''Saves a new configuration file to the path specified'''
    output = join(fpath, 'moseq-default-config.yaml')
    with open(output, 'w') as f:
        config = create_config()
        yaml.dump(config, f)
    return output


def flatten_config(config: Dict) -> Dict:
    '''Takes each sub-dictionary from the config file and makes it into one dict
    '''
    return {**config['extract'], **config['cables'], **config['background']}

def merge_cli_config(config_cli, config_file) -> Dict:
    '''Merge the keys and values from both the config file and cli options.
    This function will prefer config options, overwriting config file options.
    Returns:
        a dict with merged parameters from the config file and cli
    '''
    merged = {}
    # makes it easy for comparison
    config_file = flatten_config(config_file)
    # go through all keys found in both
    for k in set(list(config_file.keys())+list(config_cli.keys())):
        cli_val = config_cli.get(k, None)
        if cli_val is None or not cli_val:
            merged[k] = config_file[k]
        else:
            merged[k] = config_cli[k]
    return merged
