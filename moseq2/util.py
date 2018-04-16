import numpy as np
import os
import json
import cv2
import ruamel.yaml as yaml
import re
import datetime
import h5py


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


def recursive_find_h5s(root_dir=os.getcwd(),
                       ext='.h5',
                       yaml_string='{}.yaml'):
    """Recursively find h5 files, along with yaml files with the same basename
    """
    dicts = []
    h5s = []
    yamls = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            yaml_file = yaml_string.format(os.path.splitext(file)[0])
            if file.endswith(ext) and os.path.exists(os.path.join(root, yaml_file)):
                h5s.append(os.path.join(root, file))
                yamls.append(os.path.join(root, yaml_file))
                with open(os.path.join(root, yaml_file), 'r') as f:
                    dicts.append(yaml.load(f.read(), Loader=yaml.Loader))

    return h5s, dicts, yamls


# https://gist.github.com/jaytaylor/3660565
_underscorer1 = re.compile(r'(.)([A-Z][a-z]+)')
_underscorer2 = re.compile('([a-z0-9])([A-Z])')


def camel_to_snake(s):
    """Converts CamelCase to snake_case
    """
    subbed = _underscorer1.sub(r'\1_\2', s)
    return _underscorer2.sub(r'\1_\2', subbed).lower()


def build_path(key_dict, path_string, filter_spaces='-', snake_case=True, compact_time=True):
    """Takes our path string and replaces variables surrounded by braces and prefixed by $
    with a particular value in a key dictionary

    Args:
        key_dict: dictionary where each key, value pair corresponds to a variable and its value
        path_string: path string that specifies how to build our target path_string
        filter_spaces: replaces spaces with the supplied string
        snake_case: converts CamelCase to snake_case
        compact_time: converts the timestamp stored in the metadata.json file with a more compact form
    Returns:
        path_string: new path to use

    For example, if the path_string is ${root}/${subject} and key_dict is {'root':'cool','subject':'15781'}
    the path_string is converted to cool/15781
    """
    for key, value in key_dict.items():
        if key == "start_time" and compact_time:
            tmp = datetime.datetime.strptime(value.split('.')[0], "%Y-%m-%dT%H:%M:%S")
            value = tmp.strftime('%Y-%m-%dT%H:%M:%S')
        path_string = re.sub('\$\{'+key+'\}', value, path_string)

    if filter_spaces is not None:
        path_string = re.sub(' ', filter_spaces, path_string)

    if snake_case:
        path_string = camel_to_snake(path_string)

    return path_string


def h5_to_dict(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = h5_to_dict(h5file, path + key + '/')
    return ans
