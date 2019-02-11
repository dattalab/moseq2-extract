import numpy as np
import os
import json
import cv2
import click
import ruamel.yaml as yaml
import h5py

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
