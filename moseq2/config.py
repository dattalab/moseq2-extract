'''
Author: @wingillis
Module to handle operations dealing with moseq configurations.
There are two ways to define moseq parameters:
    - using the cli-based flags
    - using a configuration file
Any cli-based flags will override parameters defined in a config file
'''
import yaml
from typing import Dict

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


def load_config():
    pass


def save_config():
    pass
