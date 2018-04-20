import numpy as np
import os
import json
import cv2
import click
import ruamel.yaml as yaml


# from https://stackoverflow.com/questions/46358797/
# python-click-supply-arguments-and-options-from-a-configuration-file
def command_with_config(config_file_param_name):

    class custom_command_class(click.Command):

        def invoke(self, ctx):
            config_file = ctx.params[config_file_param_name]
            param_defaults = {}
            for param in self.params:
                if type(param) is click.core.Option:
                    param_defaults[param.human_readable_name] = param.default

            if config_file is not None:
                with open(config_file) as f:
                    config_data = yaml.load(f, yaml.RoundTripLoader)
                    for param, value in ctx.params.items():
                        if param in config_data:
                            if type(value) is tuple and type(config_data[param]) is int:
                                ctx.params[param] = tuple([config_data[param]])
                            elif type(value) is tuple:
                                ctx.params[param] = tuple(config_data[param])
                            else:
                                ctx.params[param] = config_data[param]

                            if param_defaults[param] != value:
                                ctx.params[param] = value

            return super(custom_command_class, self).invoke(ctx)

    return custom_command_class


def gen_batch_sequence(nframes, chunk_size, overlap):
    """Generate a sequence with overlap
    """
    seq = range(nframes)
    for i in range(0, len(seq)-overlap, chunk_size-overlap):
        yield seq[i:i+chunk_size]


def load_timestamps(timestamp_file, col=0):
    """Read timestamps from space delimited text file
    """

    ts = []
    with open(timestamp_file, 'r') as f:
        for line in f:
            cols = line.split()
            ts.append(float(cols[col]))

    return np.array(ts)


def load_metadata(metadata_file):

    metadata = {}

    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    return metadata


def select_strel(string='e', size=(10, 10)):
    if string[0].lower() == 'e':
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif string[0].lower() == 'r':
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    return strel
