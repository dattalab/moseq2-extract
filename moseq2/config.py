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

class InvalidConfiguration(Exception):
    pass

def create_config() -> Dict:
    '''Generate a new configuration file with all parameters set at their
    default values.
    Returns:
        A dict with default configuration values
    '''
    background = {
        'dilate': (10, 10), # how much to dilate the environment (env) floor to include walls
        'shape': 'ellipse', # shape of the floor dilation (rect for square envs and ellipse for circle envs)
        'index': 0, # if there are multiple envs in one recording, select which roi to use here
        'feature-weights': (1, .1, 1), # (area, extent, distance) Which features matter most for background selection
        'use-plane': False # if the mouse does not move a lot, use a plane instead of the background ROI
    }
    extract = {
        'min-height': 10, # minimum height of mouse from floor (mm)
        'max-height': 100,
        'fps': 30,
        'flip-classifier': None, # filepath for the flip classifier
        'chunk-size': 1000, # 1000 frames per chunk to be processed
        'chunk-overlap': 60, # 60 frames of overlap per chunk
        'cable-extraction': False, # extract data with a cable in it (i.e. use em tracking)
        'write-movie-output': True, # write movie of the extracted mouse results into an mp4
        'prefilter-time': tuple(), # a kernel to filter the mouse temporally
        'prefilter-space': (3, ) # a kernel to filter the mouse spatially
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
    keys = list(config.keys())

    # test to make sure all param types are present
    test_keys = list(create_config().keys())
    for k in test_keys:
        if k not in keys:
            raise InvalidConfiguration('Configuration file does not contain necessary keys: {}'
                                       .format(', '.join(test_keys)))

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
