'''
General helper/convenience utilities that are implemented throughout the extract package.
'''
import os
import re
import cv2
import math
import json
import h5py
import click
import warnings
import numpy as np
from glob import glob
from copy import deepcopy
import ruamel.yaml as yaml
from typing import Pattern
from cytoolz import valmap
from moseq2_extract.io.image import write_image
from os.path import join, exists, splitext, basename, abspath


def filter_warnings(func):
    def apply_warning_filters(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
            warnings.simplefilter(action='ignore', category=FutureWarning)
            warnings.simplefilter(action='ignore', category=UserWarning)
            return func(*args, **kwargs)
    return apply_warning_filters


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

                config_data = read_yaml(config_file)
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

def set_bground_to_plane_fit(bground_im, plane, output_dir):
    '''
    Replaces median-computed background image with plane fit.
    Only occurs if config_data['use_plane_bground'] == True.

    Parameters
    ----------
    bground_im (2D numpy array): Background image computed via median value of depth video.
    plane (2D numpy array): Computed ROI Plane using RANSAC.
    output_dir (str): Path to write updated background image to.

    Returns
    -------
    bground_im (2D numpy array): Plane fit version of the background image.
    '''

    xx, yy = np.meshgrid(np.arange(bground_im.shape[1]), np.arange(bground_im.shape[0]))
    coords = np.vstack((xx.ravel(), yy.ravel()))

    plane_im = (np.dot(coords.T, plane[:2]) + plane[3]) / -plane[2]
    plane_im = plane_im.reshape(bground_im.shape)

    write_image(join(output_dir, 'bground.tiff'), plane_im, scale=True)

    return plane_im

def get_frame_range_indices(config_data, nframes):
    '''
    Computes the total number of frames to be extracted, given the total number of frames
    and an initial frame index starting point.

    Parameters
    ----------
    config_data (dict): dictionary holding all extraction parameters
    nframes (int): total number of requested frames to extract

    Returns
    -------
    nframes (int): total number of frames to extract
    first_frame_idx (int): index of the frame to begin extraction from
    last_frame_idx (int): index of the last frame in the extraction
    '''

    if config_data['frame_trim'][0] > 0 and config_data['frame_trim'][0] < nframes:
        first_frame_idx = config_data['frame_trim'][0]
    else:
        first_frame_idx = 0

    if nframes - config_data['frame_trim'][1] > first_frame_idx:
        last_frame_idx = nframes - config_data['frame_trim'][1]
    else:
        last_frame_idx = nframes

    nframes = last_frame_idx - first_frame_idx

    return nframes, first_frame_idx, last_frame_idx

def gen_batch_sequence(nframes, chunk_size, overlap, offset=0):
    '''
    Generates batches used to chunk videos prior to extraction.

    Parameters
    ----------
    nframes (int): total number of frames
    chunk_size (int): desired chunk size
    overlap (int): number of overlapping frames
    offset (int): frame offset

    Returns
    -------
    Yields list of batches
    '''

    seq = range(offset, nframes)
    for i in range(offset, len(seq) - overlap, chunk_size - overlap):
        yield seq[i:i + chunk_size]


def load_timestamps(timestamp_file, col=0, alternate=False):
    '''
    Read timestamps from space delimited text file.

    Parameters
    ----------
    timestamp_file (str): path to timestamp file
    col (int): column in ts file read.
    alternate (boolean): flag set to true if timestamps were saved in a csv file

    Returns
    -------
    ts (1D array): list of timestamps
    '''

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
        warnings.warn('Timestamp file was not found! Make sure the timestamp file exists is named \
            "depth_ts.txt" or "timestamps.csv".')
        warnings.warn('This could cause issues for large number of dropped frames during the PCA step while \
            imputing missing data.')

    # if timestamps were saved in a csv file
    if alternate:
        ts = ts * 1000

    return ts

def set_bg_roi_weights(config_data):
    '''
    Reads any inputted camera type and sets the bg_roi_weights to some precomputed values.
    If no camera_type is inputted, program will assume a kinect camera is being used.

    Parameters
    ----------
    config_data (dict): dictionary containing all input parameters to a CLI/GUI command.

    Returns
    -------
    config_data (dict): updated dictionary with bg-roi-weights to use in extraction/ROI retrieval.
    '''

    # Auto-setting background weights
    camera_type = config_data.get('camera_type')
    if camera_type == 'kinect':
        config_data['bg_roi_weights'] = (1, .1, 1)
    elif camera_type == 'azure':
        config_data['bg_roi_weights'] = (10, 0.1, 1)
    elif camera_type == 'realsense':
        config_data['bg_roi_weights'] = (10, 1, 4)
    else:
        warnings.warn('Using default bg-roi-weights: (1, .1, 1)')
        config_data['bg_roi_weights'] = (1, .1, 1)

    return config_data

def check_filter_sizes(config_data):
    '''
    Checks if inputted spatial and temporal filter kernel sizes are odd numbers.
    Incrementing the value if not.

    Parameters
    ----------
    config_data (dict): Configuration dict holding all extraction parameters

    Returns
    -------
    config_data (dict): Updated configuration dict

    '''

    # Ensure filter kernel sizes are odd
    if config_data['spatial_filter_size'][0] % 2 == 0 and config_data['spatial_filter_size'][0] > 0:
        warnings.warn("Spatial Filter Size must be an odd number. Incrementing value by 1.")
        config_data['spatial_filter_size'][0] += 1
    if config_data['temporal_filter_size'][0] % 2 == 0 and config_data['temporal_filter_size'][0] > 0:
        config_data['temporal_filter_size'][0] += 1
        warnings.warn("Spatial Filter Size must be an odd number. Incrementing value by 1.")

    return config_data

def load_metadata(metadata_file):
    '''
    Loads metadata from session metadata.json file.

    Parameters
    ----------
    metadata_file (str): path to metadata file

    Returns
    -------
    metadata (dict): key-value pairs of JSON contents
    '''

    metadata = {}

    try:
        if exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
    except TypeError:
        # try loading directly
        metadata = json.load(metadata_file)

    return metadata

def load_found_session_paths(input_dir, exts):
    '''
    Given an input directory and file extensions, this function will return all
    depth file paths found in the inputted parent (input) directory.

    Parameters
    ----------
    input_dir (str): path to parent directory holding all the session folders.
    exts (list or str): list of extensions to search for, or a single extension in string form.

    Returns
    -------
    files (list): sorted list of all paths to found depth files
    '''

    if not isinstance(exts, (tuple, list)):
        exts = [exts]

    files = []
    for ext in exts:
        files.extend(glob(join(input_dir, '*/*' + ext), recursive=True))

    return sorted(files)

def get_strels(config_data):
    '''
    Get dictionary object of cv2 StructuringElements for image filtering given
    a dict of configurations parameters.

    Parameters
    ----------
    config_data (dict): dict containing cv2 Structuring Element parameters

    Returns
    -------
    str_els (dict): dict containing cv2 StructuringElements used for image filtering
    '''

    str_els = {
        'strel_dilate': select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_dilate'])),
        'strel_erode': select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_erode'])),
        'strel_tail': select_strel(config_data['tail_filter_shape'], tuple(config_data['tail_filter_size'])),
        'strel_min': select_strel(config_data['cable_filter_shape'], tuple(config_data['cable_filter_size']))
    }

    return str_els

def select_strel(string='e', size=(10, 10)):
    '''
    Returns structuring element of specified shape.
    Accepted shapes are 'ellipse' and 'rectangle'. Otherwise, 'ellipse' will be use.

    Parameters
    ----------
    string (str): indicates whether to use ellipse or rectangle
    size (tuple): size of structuring element

    Returns
    -------
    strel (cv2.StructuringElement)
    '''

    if string[0].lower() == 'e':
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif string[0].lower() == 'r':
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    return strel

def convert_pxs_to_mm(coords, resolution=(512, 424), field_of_view=(70.6, 60), true_depth=673.1):
    '''
    Converts x, y coordinates in pixel space to mm.

    http://stackoverflow.com/questions/17832238/kinect-intrinsic-parameters-from-field-of-view/18199938#18199938
    http://www.imaginativeuniversal.com/blog/post/2014/03/05/quick-reference-kinect-1-vs-kinect-2.aspx
    http://smeenk.com/kinect-field-of-view-comparison/

    Parameters
    ----------
    coords (list): list of x,y pixel coordinates
    resolution (tuple): image dimensions
    field_of_view (tuple): width and height scaling params
    true_depth (float): detected true depth

    Returns
    -------
    new_coords (list): x,y coordinates in mm
    '''

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
    '''
    Gets scalar attributes dict with names paired with descriptions.

    Returns
    -------
    attributes (dict): collection of metadata keys and descriptions.
    '''

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
        'velocity_3d_mm': '3D velocity (mm / frame), note that missing frames are not accounted for',
        'width_mm': 'Mouse width (mm)',
        'length_mm': 'Mouse length (mm)',
        'area_mm': 'Mouse area (mm)',
        'height_ave_mm': 'Mouse average height (mm)',
        'angle': 'Angle (radians, unwrapped)',
        'velocity_theta': 'Angular component of velocity (arctan(vel_x, vel_y))'
    }

    return attributes


def convert_raw_to_avi_function(input_file, chunk_size=2000, fps=30, delete=False, threads=3):
    '''
    Converts depth file (.dat, '.mkv') to avi file.

    Parameters
    ----------
    input_file (str): path to depth file
    chunk_size (int): size of chunks to process at a time
    fps (int): frames per second
    delete (bool): whether to delete original depth file
    threads (int): number of threads to write video.

    Returns
    -------
    None
    '''

    new_file = f'{splitext(input_file)[0]}.avi'
    print(f'Converting {input_file} to {new_file}')
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
    base_command = f'moseq2-extract convert-raw-to-avi {input_file}'
    for k, v in use_kwargs.items():
        base_command += f' --{k} {v}'
    for k, v in use_flags.items():
        if v:
            base_command += f' --{k}'

    print(base_command)
    print()

    os.system(base_command)

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    '''
    from https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052
    Creates subarrays of an array, a, with a given stride and window length.

    Parameters
    ----------
    a (np.ndarray) - array to get subarrarys from.
    L (int) - Window Length
    S (int) - Stride size

    Returns
    -------
    (np.ndarray) - array of subarrays at stride S.
    '''

    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))


def dict_to_h5(h5, dic, root='/', annotations=None):
    '''
    Save an dict to an h5 file, mounting at root.
    Keys are mapped to group names recursively.

    Parameters
    ----------
    h5 (h5py.File instance): h5py.file object to operate on
    dic (dict): dictionary of data to write
    root (string): group on which to add additional groups and datasets
    annotations (dict): annotation data to add to corresponding h5 datasets. Should contain same keys as dic.

    Returns
    -------
    None
    '''

    if not root.endswith('/'):
        root = root + '/'

    if annotations is None:
        annotations = {} #empty dict is better than None, but dicts shouldn't be default parameters

    for key, item in dic.items():
        dest = root + key
        try:
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                h5[dest] = item
            elif isinstance(item, (tuple, list)):
                h5[dest] = np.asarray(item)
            elif isinstance(item, (int, float)):
                h5[dest] = np.asarray([item])[0]
            elif item is None:
                h5.create_dataset(dest, data=h5py.Empty(dtype=h5py.special_dtype(vlen=str)))
            elif isinstance(item, dict):
                dict_to_h5(h5, item, dest)
            else:
                raise ValueError('Cannot save {} type to key {}'.format(type(item), dest))
        except Exception as e:
            print(e)
            if key != 'inputs':
                print('h5py could not encode key:', key)

        if key in annotations:
            if annotations[key] is None:
                h5[dest].attrs['description'] = ""
            else:
                h5[dest].attrs['description'] = annotations[key]


def recursive_find_h5s(root_dir=os.getcwd(),
                       ext='.h5',
                       yaml_string='{}.yaml'):
    '''
    Recursively find h5 files, along with yaml files with the same basename

    Parameters
    ----------
    root_dir (str): path to base directory to begin recursive search in.
    ext (str): extension to search for
    yaml_string (str): string for filename formatting when saving data

    Returns
    -------
    h5s (list): list of found h5 files
    dicts (list): list of found metadata files
    yamls (list): list of found yaml files
    '''
    if not ext.startswith('.'):
        ext = '.' + ext

    def has_frames(f):
        try:
            with h5py.File(f, 'r') as h5f:
                return 'frames' in h5f
        except OSError:
            warnings.warn(f'Error reading {f}, skipping...')
            return False

    h5s = glob(join(abspath(root_dir), '**', f'*{ext}'), recursive=True)
    h5s = filter(lambda f: exists(yaml_string.format(f.replace(ext, ''))), h5s)
    h5s = list(filter(has_frames, h5s))
    yamls = list(map(lambda f: yaml_string.format(f.replace(ext, '')), h5s))
    dicts = list(map(read_yaml, yamls))

    return h5s, dicts, yamls


def escape_path(path):
    '''
    Given current path, will return a path to return to original base directory.
    (Used in recursive h5 search, etc.)

    Parameters
    ----------
    path (str): path to current working dir

    Returns
    -------
    path (str): path to original base_dir
    '''

    return re.sub(r'\s', '\ ', path)


def clean_file_str(file_str: str, replace_with: str = '-') -> str:
    '''
    Removes invalid characters for a file name from a string.

    Parameters
    ----------
    file_str (str): filename substring to replace
    replace_with (str): value to replace str with

    Returns
    -------
    out (str): cleaned file string
    '''

    out = re.sub(r'[ <>:"/\\|?*\']', replace_with, file_str)
    # find any occurrences of `replace_with`, i.e. (--)
    return re.sub(replace_with * 2, replace_with, out)


def load_textdata(data_file, dtype=np.float32):
    '''
    Loads timestamp from txt/csv file.
    Timestamps are separated by newlines and have a space-separated data indicator,
    (in most cases, the indicator equals 0)

    Parameters
    ----------
    data_file (str): path to timestamp file
    dtype (dtype): data type of timestamps

    Returns
    -------
    data (np.ndarray): timestamp data
    timestamps (np.array): time stamp keynames.
    '''

    data = []
    timestamps = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            tmp = line.split(' ', 1)
            # appending timestamp value
            timestamps.append(int(float(tmp[0])))

            # append data indicator value
            clean_data = np.fromstring(tmp[1].replace(" ", "").strip(), sep=',', dtype=dtype)
            data.append(clean_data)

    data = np.stack(data, axis=0).squeeze()
    timestamps = np.array(timestamps, dtype=np.int)

    return data, timestamps


def time_str_for_filename(time_str: str) -> str:
    '''
    Process the time string supplied by moseq to be used in a filename. This
    removes colons, milliseconds, and timezones.

    Parameters
    ----------
    time_str (str): time str to format

    Returns
    -------
    out (str): formatted timestamp str
    '''

    out = time_str.split('.')[0]
    out = out.replace(':', '-').replace('T', '_')
    return out

def build_path(keys: dict, format_string: str, snake_case=True) -> str:
    '''
    Produce a new file name using keys collected from extraction h5 files. The format string
    must be using python's formatting specification, i.e. '{subject_name}_{session_name}'.

    Parameters
    ----------
    keys (dict): dictionary specifying which keys used to produce the new file name
    format_string (str): the string to reformat using the `keys` dictionary
    snake_case (bool): whether to save the files with snake_case

    Returns
    -------
    out (str): a newly formatted filename useable with any operating system
    '''

    if 'start_time' in keys:
        # process the time value
        keys['start_time'] = time_str_for_filename(keys['start_time'])

    if snake_case:
        keys = valmap(camel_to_snake, keys)

    return clean_file_str(format_string.format(**keys))

def read_yaml(yaml_file):
    '''
    Reads yaml file into dict object

    Parameters
    ----------
    yaml_file (str): path to yaml file

    Returns
    -------
    return_dict (dict): dict of yaml contents
    '''

    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def mouse_threshold_filter(h5file, thresh=0):
    '''
    Filters frames in h5 files by threshold value.
     Filters out frames with a nanmean < thresh.

    Parameters
    ----------
    h5file (str): path to h5 file
    thresh (int): threshold at which to apply filter

    Returns
    -------
    (3d-np boolean array): array of regions to include after threshold filter.
    '''

    with h5py.File(h5file, 'r') as f:
        # select 1st 1000 frames
        frames = f['frames'][:min(f['frames'].shape[0], 1000)]
    return np.nanmean(frames) > thresh

def _load_h5_to_dict(file: h5py.File, path) -> dict:
    '''
    Loads h5 contents to dictionary object.

    Parameters
    ----------
    h5file (h5py.File): file path to the given h5 file or the h5 file handle
    path (str): path to the base dataset within the h5 file

    Returns
    -------
    out (dict): a dict with h5 file contents with the same path structure
    '''

    ans = {}
    for key, item in file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _load_h5_to_dict(file, '/'.join([path, key]))
    return ans


def h5_to_dict(h5file, path) -> dict:
    '''
    Loads h5 contents to dictionary object.

    Parameters
    ----------
    h5file (str or h5py.File): file path to the given h5 file or the h5 file handle
    path (str): path to the base dataset within the h5 file

    Returns
    -------
    out (dict): a dict with h5 file contents with the same path structure
    '''

    if isinstance(h5file, str):
        with h5py.File(h5file, 'r') as f:
            out = _load_h5_to_dict(f, path)
    elif isinstance(h5file, h5py.File):
        out = _load_h5_to_dict(h5file, path)
    else:
        raise Exception('file input not understood - need h5 file path or file object')
    return out

def clean_dict(dct):
    '''
    Standardizes types of dict value.

    Parameters
    ----------
    dct (dict): dict object with mixed type value objects.

    Returns
    -------
    dct (dict): dict object with list value objects.
    '''

    def clean_entry(e):
        if isinstance(e, dict):
            out = clean_dict(e)
        elif isinstance(e, np.ndarray):
            out = e.tolist()
        elif isinstance(e, np.generic):
            out = np.asscalar(e)
        else:
            out = e
        return out

    return valmap(clean_entry, dct)

_underscorer1: Pattern[str] = re.compile(r'(.)([A-Z][a-z]+)')
_underscorer2 = re.compile('([a-z0-9])([A-Z])')

def camel_to_snake(s):
    '''
    Converts CamelCase to snake_case

    Parameters
    ----------
    s (str): CamelCase string to convert to snake_case.

    Returns
    -------
    (str): string in snake_case
    '''

    subbed = _underscorer1.sub(r'\1_\2', s)
    return _underscorer2.sub(r'\1_\2', subbed).lower()


def recursive_find_unextracted_dirs(root_dir=os.getcwd(),
                                    session_pattern=r'session_\d+\.(?:tgz|tar\.gz)',
                                    extension='.dat',
                                    yaml_path='proc/results_00.yaml',
                                    metadata_path='metadata.json',
                                    skip_checks=False):
    '''
    Recursively find unextracted (or incompletely extracted) directories

    Parameters
    ----------
    root_dir (os Path-like): path to base directory to start recursive search from.
    session_pattern (str): folder name pattern to search for
    extension (str): file extension to search for
    yaml_path (str): path to respective extracted metadata
    metadata_path (str): path to relative metadata.json files
    skip_checks (bool): indicates whether to check if the files exist at the given relative paths

    Returns
    -------
    proc_dirs (1d-list): list of paths to each unextracted session's proc/ directory
    '''

    from moseq2_extract.helpers.data import check_completion_status

    session_archive_pattern = re.compile(session_pattern)

    proc_dirs = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):  # test for uncompressed session
                status_file = join(root, yaml_path)
                metadata_file = join(root, metadata_path)
            elif session_archive_pattern.fullmatch(file):  # test for compressed session
                session_name = basename(file).replace('.tar.gz', '').replace('.tgz', '')
                status_file = join(root, session_name, yaml_path)
                metadata_file = join(root, '{}.json'.format(session_name))
            else:
                continue  # skip this current file as it does not look like session data

            # perform checks, append depth file to list if extraction is missing or incomplete
            if skip_checks or (not check_completion_status(status_file) and exists(metadata_file)):
                proc_dirs.append(join(root, file))

    return proc_dirs

def click_param_annot(click_cmd):
    '''
    Given a click.Command instance, return a dict that maps option names to help strings.
    Currently skips click.Arguments, as they do not have help strings.

    Parameters
    ----------
    click_cmd (click.Command): command to introspect

    Returns
    -------
    annotations (dict): click.Option.human_readable_name as keys; click.Option.help as values
    '''

    annotations = {}
    for p in click_cmd.params:
        if isinstance(p, click.Option):
            annotations[p.human_readable_name] = p.help
    return annotations

def get_bucket_center(img, true_depth, threshold=650):
    '''
    https://stackoverflow.com/questions/19768508/python-opencv-finding-circle-sun-coordinates-of-center-the-circle-from-pictu
    Finds Centroid coordinates of circular bucket.

    Parameters
    ----------
    img (2d np.ndaarray): original background image.
    true_depth (float): distance value from camera to bucket floor (automatically pre-computed)
    threshold (float): distance values to accept region into detected circle. (used to reduce fall noise interference)

    Returns
    -------
    cX (int): x-coordinate of circle centroid
    cY (int): y-coordinate of circle centroid
    '''

    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(img, threshold, true_depth, 0)

    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY

def make_gradient(width, height, h, k, a, b, theta=0):
    '''
    https://stackoverflow.com/questions/49829783/draw-a-gradual-change-ellipse-in-skimage/49848093#49848093
    Creates gradient around bucket floor representing slanted wall values.
    This is done by drawing an "ellipse" of equal x,y radii, resulting in a circle with weighted
    depth values from highest to lowest surrounding the circumference of the circle

    Parameters
    ----------
    width (int): bounding box width
    height (int) bounding box height
    h (int): centroid x coordinate
    k (int): centroid y coordinate
    a (int): x-radius of drawn ellipse
    b (int): y-radius of drawn ellipse
    theta (float): degree to rotate ellipse in radians. (has no effect if drawing a circle)

    Returns
    -------
    (2d np.ndarray): numpy array with weighted values from 0.08 -> 0.8 representing the proportion of values
    to create a gradient from. 0.8 being the proportioned values closest to the circle wall.
    '''

    # Precalculate constants
    st, ct = math.sin(theta), math.cos(theta)
    aa, bb = a ** 2, b ** 2

    # Generate (x,y) coordinate arrays
    y, x = np.mgrid[-k:height - k, -h:width - h]

    # Calculate the weight for each pixel
    weights = (((x * ct + y * st) ** 2) / aa) + (((x * st - y * ct) ** 2) / bb)

    return np.clip(1 - weights, 0.08, 0.8)


def graduate_dilated_wall_area(bground_im, config_data, strel_dilate, output_dir):
    '''
    Creates a gradient to represent the dilated (now visible) bucket wall regions.
    Only is used if background is dilated to capture larger rodents in convex shaped buckets (\_/).
    This is done to handle noise attributed by bucket walls being slanted, and thus being picked
    up as large noise depth values. Moreover, to appropriately subtract the background from input
    images during extraction without obscuring the rodent, or including unwanted wall regions.

    Parameters
    ----------
    bground_im (2d np.ndarray): the bucket floor image computed as the median distance throughout the session.
    config_data (dict): dictionary containing helper user configuration parameters.
    strel_dilate (cv2.structuringElement): dilation structuring element used to dilate background image.
    output_dir (str): path to save newly computed background to use.

    Returns
    -------
    bground_im (2d np.ndarray): the new background image with a gradient around the floor from high to low depth values.
    '''

    # store old and new backgrounds
    old_bg = deepcopy(bground_im)

    # dilate background size to match ROI size and attribute wall noise to cancel out
    bground_im = cv2.dilate(old_bg, strel_dilate, iterations=config_data.get('dilate_iterations', 5))

    # determine center of bground roi
    width, height = bground_im.shape[1], bground_im.shape[0]  # shape of bounding box

    # getting helper user parameters
    true_depth = config_data['true_depth']
    xoffset = config_data.get('x_bg_offset', -2)
    yoffset = config_data.get('y_offset', 2)
    widen_radius = config_data.get('widen_radius', 0)
    bg_threshold = config_data.get('bg_threshold', 650)

    # getting bground centroid
    cx, cy = get_bucket_center(deepcopy(old_bg), true_depth, threshold=bg_threshold)

    # set up gradient
    h, k = cx + xoffset, cy + yoffset   # centroid of gradient circle
    a, b = cx + widen_radius + 67, cy + widen_radius + 67 # x,y radii of gradient circle
    theta = math.pi/24 # gradient angle; arbitrary - used to rotate ellipses.

    # create slant gradient
    bground_im = np.float64((make_gradient(width, height, h, k, a, b, theta)) * 255)

    # scale it back to depth
    bground_im *= np.uint8((true_depth*1.1) / (bground_im.max()))  # fine-tuned - probably needs revising

    # overlay with actual bucket floor distance
    mask = np.ma.equal(old_bg, old_bg.max())
    bground_im = np.where(mask == True, old_bg, bground_im)
    bground_im = cv2.GaussianBlur(bground_im, (7, 7), 7)

    write_image(join(output_dir, 'new_bg.tiff'), bground_im, scale=True)

    return bground_im