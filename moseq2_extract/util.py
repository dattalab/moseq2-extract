import numpy as np
import os
import json
import cv2
import click
import ruamel.yaml as yaml
import h5py
import re
from typing import Pattern
import warnings
from cytoolz import valmap, assoc

# from https://stackoverflow.com/questions/46358797/
# python-click-supply-arguments-and-options-from-a-configuration-file
def command_with_config(config_file_param_name):

    class custom_command_class(click.Command):

        def invoke(self, ctx):
            # grab the config file
            config_file = ctx.params[config_file_param_name]
            param_defaults = {p.human_readable_name: p.default for p in self.params
                              if isinstance(p, click.core.Option)}
            param_defaults = {k: tuple(v) if type(v) is list else v for k, v in param_defaults.items()}
            param_cli = {k: tuple(v) if type(v) is list else v for k, v in ctx.params.items()}

            if config_file is not None:
                with open(config_file) as f:
                    config_data = dict(yaml.load(f, yaml.RoundTripLoader))
                # modified to only use keys that are actually defined in options
                config_data = {k: tuple(v) if isinstance(v, yaml.comments.CommentedSeq) else v
                               for k, v in config_data.items() if k in param_defaults.keys()}

                # find differences btw config and param defaults
                diffs = set(param_defaults.items()) ^ set(param_cli.items())

                # combine defaults w/ config data
                combined = {**param_defaults, **config_data}

                # update cli params that are non-default
                keys = [d[0] for d in diffs]
                for k in set(keys):
                    combined[k] = ctx.params[k]

                ctx.params = combined

            return super().invoke(ctx)

    return custom_command_class


def gen_batch_sequence(nframes, chunk_size, overlap, offset=0):
    """Generate a sequence with overlap
    """
    seq = range(offset, nframes)
    for i in range(offset, len(seq)-overlap, chunk_size-overlap):
        yield seq[i:i+chunk_size]


def load_timestamps(timestamp_file, col=0):
    """Read timestamps from space delimited text file
    """

    ts = []
    try:
        with open(timestamp_file, 'r') as f:
            for line in f:
                cols = line.split()
                ts.append(float(cols[col]))
        ts = np.array(ts)
    except TypeError as e:
        # try iterating directly
        for line in timestamp_file:
            cols = line.split()
            ts.append(float(cols[col]))
        ts = np.array(ts)
    except FileNotFoundError as e:
        ts = None

    return ts


def load_metadata(metadata_file):

    metadata = {}

    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
    except TypeError as e:
        # try loading directly
        metadata = json.load(metadata_file)

    return metadata


def select_strel(string='e', size=(10, 10)):
    if string[0].lower() == 'e':
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif string[0].lower() == 'r':
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    return strel


# http://stackoverflow.com/questions/17832238/kinect-intrinsic-parameters-from-field-of-view/18199938#18199938
# http://www.imaginativeuniversal.com/blog/post/2014/03/05/quick-reference-kinect-1-vs-kinect-2.aspx
# http://smeenk.com/kinect-field-of-view-comparison/
def convert_pxs_to_mm(coords, resolution=(512, 424), field_of_view=(70.6, 60), true_depth=673.1):
    """Converts x, y coordinates in pixel space to mm
    """
    cx = resolution[0] // 2
    cy = resolution[1] // 2

    xhat = coords[:, 0] - cx
    yhat = coords[:, 1] - cy

    fw = resolution[0] / (2 * np.deg2rad(field_of_view[0] / 2))
    fh = resolution[1] / (2 * np.deg2rad(field_of_view[1] / 2))

    new_coords = np.zeros_like(coords)
    new_coords[:, 0] = true_depth * xhat / fw
    new_coords[:, 1] = true_depth * yhat / fh

    return new_coords


def scalar_attributes():

    attributes = {
        'centroid_x_px': 'X centroid (pixels)',
        'centroid_y_px': 'Y centroid (pixels)',
        'velocity_2d_px': '2D velocity (pixels / frame), note that missing frames are not accounted for',
        'velocity_3d_px': '3D velocity (pixels / frame), note that missing frames are not accounted for, also height is in mm, not pixels for calculation',
        'width_px': 'Mouse width (pixels)',
        'length_px': 'Mouse length (pixels)',
        'area_px': 'Mouse area (pixels)',
        'centroid_x_mm': 'X centroid (mm)',
        'centroid_y_mm': 'Y centroid (mm)',
        'velocity_2d_mm': '2D velocity (mm / frame), note that missing frames are not accounted for',
        'velocity_3d_mm': '2D velocity (mm / frame), note that missing frames are not accounted for',
        'width_mm': 'Mouse width (mm)',
        'length_mm': 'Mouse length (mm)',
        'area_mm': 'Mouse area (mm)',
        'height_ave_mm': 'Mouse average height (mm)',
        'angle': 'Angle (radians, unwrapped)',
        'velocity_theta': 'Angular component of velocity (arctan(vel_x, vel_y))'
    }

    return attributes


def convert_raw_to_avi_function(input_file, chunk_size=2000, fps=30, delete=False, threads=3):

    new_file = '{}.avi'.format(os.path.splitext(input_file)[0])
    print('Converting {} to {}'.format(input_file, new_file))
    # turn into os system call...
    use_kwargs = {
        'output-file': new_file,
        'chunk-size': chunk_size,
        'fps': fps,
        'threads': threads
    }
    use_flags = {
        'delete': delete
    }
    base_command = 'moseq2-extract convert-raw-to-avi {}'.format(input_file)
    for k, v in use_kwargs.items():
        base_command += ' --{} {}'.format(k, v)
    for k, v in use_flags.items():
        if v:
            base_command += ' --{}'.format(k)

    print(base_command)
    print('\n')

    os.system(base_command)


# from https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052
# dang this is fast!
def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))


def save_dict_contents_to_h5(h5, dic, root='/', annotations=None):
    """ Save an dict to an h5 file, mounting at root

    Keys are mapped to group names recursivly

    Parameters:
        h5 (h5py.File instance): h5py.file object to operate on
        dic (dict): dictionary of data to write
        root (string): group on which to add additional groups and datasets
        annotations (dict): annotation data to add to corresponding h5 datasets. Should contain same keys as dic.
    """
    if not root.endswith('/'):
        root = root + '/'

    if annotations is None:
        annotations = {} #empty dict is better than None, but dicts shouldn't be default parameters

    for key, item in dic.items():
        dest = root + key
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5[dest] = item
        elif isinstance(item, (tuple, list)):
            h5[dest] = np.asarray(item)
        elif isinstance(item, (int, float)):
            h5[dest] = np.asarray([item])[0]
        elif item is None:
            h5.create_dataset(dest, data=h5py.Empty(dtype=h5py.special_dtype(vlen=str)))
        elif isinstance(item, dict):
            save_dict_contents_to_h5(h5, item, dest)
        else:
            raise ValueError('Cannot save {} type to key {}'.format(type(item), dest))

        if key in annotations:
            if annotations[key] is None:
                h5[dest].attrs['description'] = ""
            else:
                h5[dest].attrs['description'] = annotations[key]


def recursive_find_h5s(root_dir=os.getcwd(),
                       ext='.h5',
                       yaml_string='{}.yaml'):
    """Recursively find h5 files, along with yaml files with the same basename
    """
    dicts = []
    h5s = []
    yamls = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            yaml_file = yaml_string.format(os.path.splitext(file)[0])
            if file.endswith(ext) and os.path.exists(os.path.join(root, yaml_file)):
                try:
                    with h5py.File(os.path.join(root, file), 'r') as f:
                        if 'frames' not in f.keys():
                            continue
                except OSError:
                    warnings.warn('Error reading {}, skipping...'.format(os.path.join(root, file)))
                    continue
                h5s.append(os.path.join(root, file))
                yamls.append(os.path.join(root, yaml_file))
                dicts.append(read_yaml(os.path.join(root, yaml_file)))

    return h5s, dicts, yamls


def escape_path(path):
    return re.sub(r'\s', '\ ', path)

def clean_file_str(file_str: str, replace_with: str = '-') -> str:
    '''removes invalid characters for a file name from a string
    '''
    out = re.sub(r'[ <>:"/\\|?*\']', replace_with, file_str)
    # find any occurrences of `replace_with`, i.e. (--)
    return re.sub(replace_with * 2, replace_with, out)

def load_textdata(data_file, dtype=np.float32):

    data = []
    timestamps = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            tmp = line.split(' ', 1)
            timestamps.append(int(tmp[0]))
            clean_data = np.fromstring(tmp[1].replace(" ", "").strip(), sep=',', dtype=dtype)
            data.append(clean_data)

    data = np.stack(data, axis=0).squeeze()
    timestamps = np.array(timestamps, dtype=np.int)

    return data, timestamps


def time_str_for_filename(time_str: str) -> str:
    '''Process the time string supplied by moseq to be used in a filename. This
    removes colons, milliseconds, and timezones.
    '''
    out = time_str.split('.')[0]
    out = out.replace(':', '-').replace('T', '_')
    return out

def build_path(keys: dict, format_string: str, snake_case=True) -> str:
    '''Produce a new file name using keys collected from extraction h5 files. The format string
    must be using python's formatting specification, i.e. '{subject_name}_{session_name}'.

    Args:
        keys (dict): dictionary specifying which keys used to produce the new file name
        format_string (str): the string to reformat using the `keys` dictionary
    Returns:
        a newly formatted filename useable with any operating system

    >>> build_path(dict(a='hello', b='world'), '{a}_{b}')
    'hello_world'
    >>> build_path(dict(a='hello', b='world'), '{a}/{b}')
    'hello-world'
    '''
    if 'start_time' in keys:
        # process the time value
        val = keys['start_time']
        keys = assoc(keys, 'start_time', time_str_for_filename(val))

    if snake_case:
        keys = valmap(camel_to_snake, keys)

    return clean_file_str(format_string.format(**keys))

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        dat = f.read()
        try:
            return_dict = yaml.safe_load(dat)
        except AttributeError:
            return_dict = yaml.safe_load(dat)

    return return_dict

def mouse_threshold_filter(h5file, thresh=0):
    with h5py.File(h5file, 'r') as f:
        # select 1st 1000 frames
        frames = f['frames'][:min(f['frames'].shape[0], 1000)]
    return np.nanmean(frames) > thresh

def _load_h5_to_dict(file: h5py.File, path) -> dict:
    ans = {}
    for key, item in file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _load_h5_to_dict(file, '/'.join([path, key]))
    return ans


def h5_to_dict(h5file, path) -> dict:
    """
    Args:
        h5file (str or h5py.File): file path to the given h5 file or the h5 file handle
        path: path to the base dataset within the h5 file
    Returns:
        a dict with h5 file contents with the same path structure
    """
    if isinstance(h5file, str):
        with h5py.File(h5file, 'r') as f:
            out = _load_h5_to_dict(f, path)
    elif isinstance(h5file, h5py.File):
        out = _load_h5_to_dict(h5file, path)
    else:
        raise Exception('file input not understood - need h5 file path or file object')
    return out


_underscorer1: Pattern[str] = re.compile(r'(.)([A-Z][a-z]+)')
_underscorer2 = re.compile('([a-z0-9])([A-Z])')

def camel_to_snake(s):
    """Converts CamelCase to snake_case
    """
    subbed = _underscorer1.sub(r'\1_\2', s)
    return _underscorer2.sub(r'\1_\2', subbed).lower()


def recursive_find_unextracted_dirs(root_dir=os.getcwd(),
                                    session_pattern=r'session_\d+\.(?:tgz|tar\.gz)',
                                    filename='.dat',
                                    yaml_path='proc/results_00.yaml',
                                    metadata_path='metadata.json',
                                    skip_checks=True):
    """Recursively find unextracted directories
    """
    session_archive_pattern = re.compile(session_pattern)

    proc_dirs = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if filename in file:  # test for uncompressed session
                status_file = os.path.join(root, yaml_path)
                metadata_file = os.path.join(root, metadata_path)

            elif session_archive_pattern.fullmatch(file):  # test for compressed session
                session_name = os.path.basename(file).replace('.tar.gz', '').replace('.tgz', '')
                status_file = os.path.join(root, session_name, yaml_path)
                metadata_file = os.path.join(root, '{}.json'.format(session_name))
            else:
                continue  # skip this current file as it does not look like session data

            # perform checks
            if skip_checks or (not os.path.exists(status_file) and os.path.exists(metadata_file)):
                proc_dirs.append(os.path.join(root, file))

    return proc_dirs


def click_param_annot(click_cmd):
    """ Given a click.Command instance, return a dict that maps option names to help strings

    Currently skips click.Arguments, as they do not have help strings

    Parameters:
        click_cmd (click.Command): command to introspect

    Returns:
        dict: click.Option.human_readable_name as keys; click.Option.help as values
    """
    annotations = {}
    for p in click_cmd.params:
        if isinstance(p, click.Option):
            annotations[p.human_readable_name] = p.help
    return annotations
