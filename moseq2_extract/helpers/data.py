import os
import h5py
from tqdm.auto import tqdm
import shutil
import tarfile
import numpy as np
import ruamel.yaml as yaml
from cytoolz import keymap
from cytoolz.curried import valmap
from pkg_resources import get_distribution
from moseq2_extract.util import h5_to_dict, load_timestamps, camel_to_snake, \
    load_textdata, build_path, save_dict_contents_to_h5, click_param_annot

# extract all helper function
def get_selected_sessions(to_extract, extract_all):
    '''
    Given user input, the function will return either selected sessions to extract, or all the sessions.

    Parameters
    ----------
    to_extract (list): list of paths to sessions to extract
    extract_all (bool): boolean to include all sessions and skip user-input prompt.

    Returns
    -------
    to_extract (list): new list of selected sessions to extract.
    '''
    if len(to_extract) > 1 and not extract_all:
        for i, sess in enumerate(to_extract):
            print(f'[{str(i + 1)}] {sess}')

        input_file_indices = []
        while (True):
            try:
                input_file_indices = input("Input session indices to extract separated by commas, or empty string for all sessions.\nTo extract all except certain sessions: type 'e index1,index2,...'\n").strip()
                if 'e' not in input_file_indices:
                    if ',' in input_file_indices:
                        input_file_indices = input_file_indices.split(',')
                        for i in input_file_indices:
                            i = int(i.strip())
                            if i > len(to_extract):
                                print('invalid index try again.')
                                input_file_index = []
                                break

                        tmp = []
                        for index in input_file_indices:
                            index = int(index.strip())
                            print('extracting ', to_extract[index-1])
                            tmp.append(to_extract[index-1])
                        to_extract = tmp
                        break
                    elif len(input_file_indices.strip()) == 1:
                        index = int(input_file_indices.strip())
                        to_extract = [to_extract[index-1]]
                        print('extracting ', to_extract)
                        break
                    elif input_file_indices == '':
                        break
                else:
                    input_file_indices = input_file_indices.strip('e ')
                    if ',' in input_file_indices:
                        input_file_indices = input_file_indices.split(',')
                        for i in input_file_indices:
                            i = int(i.strip())
                            if i > len(to_extract):
                                print('invalid index try again.')
                                input_file_index = []
                                break
                        # values to remove
                        vals = []
                        for index in input_file_indices:
                            index = int(index.strip())
                            print('excluding ', to_extract[index-1])
                            vals.append(to_extract[index-1])
                        for val in vals:
                            to_extract.remove(val)
                        break
                    elif len(input_file_indices.strip()) == 1:
                        index = int(input_file_indices.strip())
                        print('excluding ', to_extract[index-1])
                        to_extract.remove(to_extract[index-1])
                        break
                    elif input_file_indices == '':
                        break

            except:
                print('invalid input, only input correct comma separated indices, or no empty string for all sessions.')
    else:
        print(f'Extracting {to_extract[0]}')

    return to_extract

#
def load_h5s(to_load, snake_case=True):
    '''
    aggregate_results() Helper Function to load h5 files.

    Parameters
    ----------
    to_load (list): list of paths to h5 files.
    snake_case (bool): whether to save the files using snake_case

    Returns
    -------
    loaded (list): list of loaded h5 dicts.
    '''
    loaded = []
    for _dict, _h5f in tqdm(to_load, desc='Scanning data'):
        try:
            # v0.1.3 introduced a change - acq. metadata now here
            tmp = h5_to_dict(_h5f, '/metadata/acquisition')
        except KeyError:
            # if it doesn't exist it's likely from an older moseq version. Try loading it here
            try:
                tmp = h5_to_dict(_h5f, '/metadata/extraction')
            except KeyError:
                # if all else fails, abandon all hope
                tmp = {}

        # note that everything going into here must be a string (no bytes!)
        tmp = {k: str(v) for k, v in tmp.items()}
        if snake_case:
            tmp = keymap(camel_to_snake, tmp)

        feedback_file = os.path.join(os.path.dirname(_h5f), '..', 'feedback_ts.txt')
        if os.path.exists(feedback_file):
            timestamps = map(int, load_timestamps(feedback_file, 0))
            feedback_status = map(int, load_timestamps(feedback_file, 1))
            _dict['feedback_timestamps'] = list(zip(timestamps, feedback_status))

        _dict['extraction_metadata'] = tmp
        loaded += [(_dict, _h5f)]

    return loaded


def build_manifest(loaded, format, snake_case=True):
    '''
    aggregate_results() Helper Function.
    Builds a manifest file used to contain extraction result metadata from h5 and yaml files.

    Parameters
    ----------
    loaded (list of dicts): list of dicts containing loaded h5 data.
    format (str): filename format indicating the new name for the metadata files in the aggregate_results dir.
    snake_case (bool): whether to save the files using snake_case

    Returns
    -------
    manifest (dict): dictionary of extraction metadata.
    '''

    manifest = {}
    fallback = 'session_{:03d}'
    fallback_count = 0

    # you know, bonus internal only stuff for the time being...
    additional_meta = []
    additional_meta.append({
        'filename': 'feedback_ts.txt',
        'var_name': 'realtime_feedback',
        'dtype': np.bool,
    })
    additional_meta.append({
        'filename': 'predictions.txt',
        'var_name': 'realtime_predictions',
        'dtype': np.int,
    })
    additional_meta.append({
        'filename': 'pc_scores.txt',
        'var_name': 'realtime_pc_scores',
        'dtype': np.float32,
    })

    for _dict, _h5f in loaded:
        print_format = '{}_{}'.format(
            format, os.path.splitext(os.path.basename(_h5f))[0])
        if not _dict['extraction_metadata']:
            copy_path = fallback.format(fallback_count)
            fallback_count += 1
        else:
            try:
                copy_path = build_path(_dict['extraction_metadata'], print_format, snake_case=snake_case)
            except:
                copy_path = fallback.format(fallback_count)
                fallback_count += 1
                pass
        # add a bonus dictionary here to be copied to h5 file itself
        manifest[_h5f] = {'copy_path': copy_path, 'yaml_dict': _dict, 'additional_metadata': {}}
        for meta in additional_meta:
            filename = os.path.join(os.path.dirname(_h5f), '..', meta['filename'])
            if os.path.exists(filename):
                try:
                    data, timestamps = load_textdata(filename, dtype=meta['dtype'])
                    manifest[_h5f]['additional_metadata'][meta['var_name']] = {
                        'data': data,
                        'timestamps': timestamps
                    }
                except:
                    print('Did not load timestamps')

    return manifest


def copy_manifest_results(manifest, output_dir):
    '''
    Copies all considated manifest results to their respective output files.

    Parameters
    ----------
    manifest (dict): manifest dictionary containing all extraction h5 metadata to save
    output_dir (str): path to directory where extraction results will be aggregated.

    Returns
    -------
    None
    '''

    # now the key is the source h5 file and the value is the path to copy to
    for k, v in tqdm(manifest.items(), desc='Copying files'):

        if os.path.exists(os.path.join(output_dir, '{}.h5'.format(v['copy_path']))):
            continue

        basename = os.path.splitext(os.path.basename(k))[0]
        dirname = os.path.dirname(k)

        h5_path = k
        mp4_path = os.path.join(dirname, '{}.mp4'.format(basename))
        # yaml_path = os.path.join(dirname, '{}.yaml'.format(basename))

        if os.path.exists(h5_path):
            new_h5_path = os.path.join(output_dir, '{}.h5'.format(v['copy_path']))
            shutil.copyfile(h5_path, new_h5_path)

        # if we have additional_meta then crack open the h5py and write to a safe place
        if len(v['additional_metadata']) > 0:
            for k2, v2 in v['additional_metadata'].items():
                new_key = '/metadata/misc/{}'.format(k2)
                with h5py.File(new_h5_path, "a") as f:
                    f.create_dataset('{}/data'.format(new_key), data=v2["data"])
                    f.create_dataset('{}/timestamps'.format(new_key), data=v2["timestamps"])

        if os.path.exists(mp4_path):
            shutil.copyfile(mp4_path, os.path.join(
                output_dir, '{}.mp4'.format(v['copy_path'])))

        v['yaml_dict'].pop('extraction_metadata', None)
        with open('{}.yaml'.format(os.path.join(output_dir, v['copy_path'])), 'w') as f:
            yaml.safe_dump(v['yaml_dict'], f)


def handle_extract_metadata(input_file, dirname, config_data, nframes):
    '''
    Extracts metadata from input depth files, either raw or compressed.

    Parameters
    ----------
    input_file (str): path to input file to extract
    dirname (str): path to directory where extraction files reside.
    config_data (dict): dictionary object containing all required extraction parameters. (auto generated)
    nframes (int): number of frames to extract.

    Returns
    -------
    metadata_path (str): path to respective metadata.json
    timestamp_path (str): path to respective depth_ts.txt or similar
    alternate_correct (bool): indicator for whether an alternate timestamp file was used
    tar (bool): indicator for whether the file is compressed.
    nframes (int): number of frames to extract
    first_frame_idx (int): index number of first frame in extraction.
    last_frame_idx (int): index number of last frame in extraction
    '''

    if input_file.endswith('.tar.gz') or input_file.endswith('.tgz'):
        print('Scanning tarball {} (this will take a minute)'.format(input_file))
        # compute NEW psuedo-dirname now, `input_file` gets overwritten below with test_vid.dat tarinfo...
        dirname = os.path.join(dirname, os.path.basename(input_file).replace('.tar.gz', '').replace('.tgz', ''))

        tar = tarfile.open(input_file, 'r:gz')
        tar_members = tar.getmembers()
        tar_names = [_.name for _ in tar_members]
        input_file = tar_members[tar_names.index('test_vid.dat')]
    else:
        tar = None
        tar_members = None

    if config_data['frame_trim'][0] > 0 and config_data['frame_trim'][0] < nframes:
        first_frame_idx = config_data['frame_trim'][0]
    else:
        first_frame_idx = 0

    if nframes - config_data['frame_trim'][1] > first_frame_idx:
        last_frame_idx = nframes - config_data['frame_trim'][1]
    else:
        last_frame_idx = nframes

    nframes = last_frame_idx - first_frame_idx
    alternate_correct = False

    if tar is not None:
        metadata_path = tar.extractfile(tar_members[tar_names.index('metadata.json')])
        if "depth_ts.txt" in tar_names:
            timestamp_path = tar.extractfile(tar_members[tar_names.index('depth_ts.txt')])
        elif "timestamps.csv" in tar_names:
            timestamp_path = tar.extractfile(tar_members[tar_names.index('timestamps.csv')])
            alternate_correct = True
    else:
        metadata_path = os.path.join(dirname, 'metadata.json')
        timestamp_path = os.path.join(dirname, 'depth_ts.txt')
        alternate_timestamp_path = os.path.join(dirname, 'timestamps.csv')
        if not os.path.exists(timestamp_path) and os.path.exists(alternate_timestamp_path):
            timestamp_path = alternate_timestamp_path
            alternate_correct = True

    return metadata_path, timestamp_path, alternate_correct, tar, nframes, first_frame_idx, last_frame_idx


# extract h5 helper function
def create_extract_h5(f, acquisition_metadata, config_data, status_dict, scalars, scalars_attrs,
                      nframes, true_depth, roi, bground_im, first_frame, timestamps, extract=None):
    '''
    Creates h5 file that holds all extracted frames and other metadata (such as scalars).

    Parameters
    ----------
    f (h5py.File object): opened h5 file object to write to.
    acquisition_metadata (dict): Dictionary containing extracted session acquisition metadata.
    config_data (dict): dictionary object containing all required extraction parameters. (auto generated)
    status_dict (dict): dictionary that helps indicate if the session has been extracted fully.
    scalars (list): list of computed scalar metadata.
    scalars_attrs (dict): dict of respective computed scalar attributes and descriptions to save.
    nframes (int): number of frames being recorded
    true_depth (float): computed detected true depth
    roi (2d np.ndarray): Computed 2D ROI Image.
    bground_im (2d np.ndarray): Computed 2D Background Image.
    first_frame (2d np.ndarray): Computed 2D First Frame Image.
    timestamps (np.array): Array of session timestamps.
    extract (moseq2_extract.cli.extract function): Used to preseve CLI state parameters in extraction h5.

    Returns
    -------
    None
    '''

    f.create_dataset('metadata/uuid', data=status_dict['uuid'])
    for scalar in scalars:
        f.create_dataset('scalars/{}'.format(scalar), (nframes,), 'float32', compression='gzip')
        f['scalars/{}'.format(scalar)].attrs['description'] = scalars_attrs[scalar]

    if timestamps is not None:
        f.create_dataset('timestamps', compression='gzip', data=timestamps)
        f['timestamps'].attrs['description'] = "Depth video timestamps"

    f.create_dataset('frames', (nframes, config_data['crop_size'][0], config_data['crop_size'][1]),
                     config_data['frame_dtype'], compression='gzip')
    f['frames'].attrs['description'] = '3D Numpy array of depth frames (nframes x w x h, in mm)'

    if config_data['use_tracking_model']:
        f.create_dataset('frames_mask', (nframes, config_data['crop_size'][0], config_data['crop_size'][1]), 'float32',
                         compression='gzip')
        f['frames_mask'].attrs['description'] = 'Log-likelihood values from the tracking model (nframes x w x h)'
    else:
        f.create_dataset('frames_mask', (nframes, config_data['crop_size'][0], config_data['crop_size'][1]), 'bool',
                         compression='gzip')
        f['frames_mask'].attrs['description'] = 'Boolean mask, false=not mouse, true=mouse'

    if config_data['flip_classifier'] is not None:
        f.create_dataset('metadata/extraction/flips', (nframes,), 'bool', compression='gzip')
        f['metadata/extraction/flips'].attrs['description'] = 'Output from flip classifier, false=no flip, true=flip'

    f.create_dataset('metadata/extraction/true_depth', data=true_depth)
    f['metadata/extraction/true_depth'].attrs['description'] = 'Detected true depth of arena floor in mm'

    f.create_dataset('metadata/extraction/roi', data=roi, compression='gzip')
    f['metadata/extraction/roi'].attrs['description'] = 'ROI mask'

    f.create_dataset('metadata/extraction/first_frame', data=first_frame[0], compression='gzip')
    f['metadata/extraction/first_frame'].attrs['description'] = 'First frame of depth dataset'

    f.create_dataset('metadata/extraction/background', data=bground_im, compression='gzip')
    f['metadata/extraction/background'].attrs['description'] = 'Computed background image'

    extract_version = np.string_(get_distribution('moseq2-extract').version)
    f.create_dataset('metadata/extraction/extract_version', data=extract_version)
    f['metadata/extraction/extract_version'].attrs['description'] = 'Version of moseq2-extract'

    if extract is not None:
        save_dict_contents_to_h5(f, status_dict['parameters'], 'metadata/extraction/parameters', click_param_annot(extract))
    else:
        save_dict_contents_to_h5(f, status_dict['parameters'], 'metadata/extraction/parameters')

    for key, value in acquisition_metadata.items():
        if type(value) is list and len(value) > 0 and type(value[0]) is str:
            value = [n.encode('utf8') for n in value]

        if value is not None:
            f.create_dataset('metadata/acquisition/{}'.format(key), data=value)
        else:
            f.create_dataset('metadata/acquisition/{}'.format(key), dtype="f")

# Viz functions -- to refactor
def _load_h5_to_dict(file: h5py.File, path: str) -> dict:
    '''
    Loads h5 file and returns dictionary object representing all contained data.

    Parameters
    ----------
    file (h5py.File): h5 file to return dict from.
    path (str): str path to data within h5 file. E.g. '/'.

    Returns
    -------
    ans (dict): dictionary object containing all requested h5 data.
    '''

    ans = {}
    if isinstance(file[path], h5py.Dataset):
        # only use the final path key to add to `ans`
        ans[path.split('/')[-1]] = file[path][()]
    else:
        for key, item in file[path].items():
            if isinstance(item, h5py.Dataset):
                ans[key] = item[()]
            elif isinstance(item, h5py.Group):
                ans[key] = _load_h5_to_dict(file, '/'.join([path, key]))
    return ans

def h5_to_dict(h5file, path: str = '/') -> dict:
    '''
    Loads h5 file and returns dictionary object representing all contained data,
    given a path within the h5 file.

    Parameters
    ----------
    h5file (str or h5py.File): file path to the given h5 file or the h5 file handle
    path (str): path to the base dataset within the h5 file. Default: '/'

    Returns
    -------
    out (dict): a dict with h5 file contents with the same path structure
    '''

    if isinstance(h5file, str):
        with h5py.File(h5file, 'r') as f:
            out = _load_h5_to_dict(f, path)
    elif isinstance(h5file, (h5py.File, h5py.Group)):
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
