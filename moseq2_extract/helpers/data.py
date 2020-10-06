'''

Data selection, writing, and loading utilities.
Contains helper functions to aid mostly in handling/storing data during extraction.
Remainder of functions are used in the data aggregation process.

'''

import os
import h5py
import json
import shutil
import tarfile
import warnings
import numpy as np
from glob import glob
import ruamel.yaml as yaml
from tqdm.auto import tqdm
from cytoolz import keymap
from ast import literal_eval
from os.path import dirname, basename
from pkg_resources import get_distribution
from moseq2_extract.util import h5_to_dict, load_timestamps, load_metadata,  \
    camel_to_snake, load_textdata, build_path, dict_to_h5, click_param_annot

# TODO: what's different about this function and recursive_find_unextracted_dirs / recursive_find_h5s
def get_session_paths(data_dir, extracted=False, exts=['dat', 'mkv', 'avi']):
    '''
    Find all depth recording sessions and their paths (with given extensions)
    to work on given base directory.

    Parameters
    ----------
    data_dir (str): path to directory containing all session folders.
    exts (list): list of depth file extensions to search for.

    Returns
    -------
    path_dict (dict): session directory name keys pair with their respective absolute paths.
    '''

    if extracted:
        path = '*/proc/*.'
        exts = ['mp4']
    else:
        path = '*/*.'

    sessions = []

    # Get list of sessions ending in the given extensions
    for ext in exts:
        if len(data_dir) == 0:
            data_dir = os.getcwd()
            files = sorted(glob(path + ext))
            sessions += files
        else:
            data_dir = data_dir.strip()
            if os.path.isdir(data_dir):
                files = sorted(glob(os.path.join(data_dir, path + ext)))
                sessions += files
            else:
                print('directory not found, try again.')

    # generate sample metadata json for each session that is missing one
    sample_meta = {'SubjectName': 'default', 'SessionName': 'default',
                   'NidaqChannels': 0, 'NidaqSamplingRate': 0.0, 'DepthResolution': [512, 424],
                   'ColorDataType': "Byte[]", "StartTime": ""}

    if not extracted:
        for sess in sessions:
            # get path to session directory
            sess_dir = dirname(sess)
            sess_name = basename(sess_dir)
            # Generate metadata.json file if it's missing
            if 'metadata.json' not in os.listdir(sess_dir):
                sample_meta['SessionName'] = sess_name
                with open(os.path.join(sess_dir, 'metadata.json'), 'w') as fp:
                    json.dump(sample_meta, fp)

        # Create path dictionary
        names = [basename(dirname(sess)) for sess in sessions]
        path_dict = {n: p for n, p in zip(names, sessions)}
    else:
        names = [dirname(sess).split('/')[-2] for sess in sessions]
        path_dict = {n: p for n, p in zip(names, sessions)}

    return path_dict

# extract_wrapper helper function
def check_completion_status(status_filename):
    '''
    Reads a results_00.yaml (status file) and checks whether the session has been
    fully extracted. Returns True if yes, and False if not and if the file doesn't exist.

    Parameters
    ----------
    status_filename (str): path to results_00.yaml containing extraction status

    Returns
    -------
    complete (bool): If True, data has been extracted to completion.
    '''

    if os.path.exists(status_filename):
        with open(status_filename, 'r') as f:
            return yaml.safe_load(f)['complete']
    return False

# extract all helper function
# TODO: this seems like it belongs in the gui.py module
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

    selected_sess_idx, excluded_sess_idx, ret_extract = [], [], []

    def parse_input(s):
        '''
        Parses user input, looking for specifically numbered sessions, ranges of sessions,
        and/or sessions to exclude.

        Function will alter the parent functions variables {selected_sess_idx, excluded_sess_idx} according to the
        user input.
        Parameters
        ----------
        s (str): User input session indices.
        Examples: "1", or "1,2,3" == "1-3", or "e 1-3" to exclude sessions 1-3.

        Returns
        -------

        '''
        if 'e' not in s and '-' not in s:
            if isinstance(literal_eval(s), int):
                selected_sess_idx.append(int(s))
        elif 'e' not in s and '-' in s:
            ss = s.split('-')
            if isinstance(literal_eval(ss[0]), int) and isinstance(literal_eval(ss[1]), int):
                for i in range(int(ss[0]), int(ss[1]) + 1):
                    selected_sess_idx.append(i)
        elif 'e' in s:
            s = s.strip('e')
            if '-' not in s:
                if isinstance(literal_eval(s), int):
                    excluded_sess_idx.append(int(s))
            else:
                ss = s.split('-')
                if isinstance(literal_eval(ss[0]), int) and isinstance(literal_eval(ss[1]), int):
                    for i in range(int(ss[0]), int(ss[1]) + 1):
                        excluded_sess_idx.append(i)

    if len(to_extract) > 1 and not extract_all:
        for i, sess in enumerate(to_extract):
            print(f'[{str(i + 1)}] {sess}')

        print('You may input comma separated values for individual sessions')
        print('Or you can input a hyphen separated range. E.g. "1-10" selects 10 sessions, including sessions 1 and 10')
        print('You can also exclude a range by prefixing the range selection with the letter "e"; e.g.: "e1-5".')
        print('Press q to quit.')
        while(len(ret_extract) == 0):
            sessions = input('Input your selected sessions to extract: ')
            if 'q' in sessions.lower():
                return []
            if ',' in sessions:
                selection = sessions.split(',')
                for s in selection:
                    s = s.strip()
                    parse_input(s)
                for i in selected_sess_idx:
                    if i not in excluded_sess_idx:
                        ret_extract.append(to_extract[i - 1])
            elif len(sessions) > 0:
                parse_input(sessions)
                for i in selected_sess_idx:
                    if i not in excluded_sess_idx:
                        if i-1 < len(to_extract):
                            ret_extract.append(to_extract[i - 1])
            else:
                print('Invalid input. Try again or press q to quit.')
    else:
        return to_extract

    return ret_extract

# TODO: Give an example of what's in the file_tup (e.g. the first entry in files_to_use) in the docs
def build_index_dict(files_to_use):
    '''
    Given a list of files and respective metadatas to include in an index file,
    creates a dictionary that will be saved later as the index file.
    It will contain all the inputted file paths with their respective uuids, group names, and metadata.

    Note: This is a direct helper function for generate_index_wrapper().

    Parameters
    ----------
    files_to_use (list): list of paths to extracted h5 files.

    Returns
    -------
    output_dict (dict): index-file dictionary containing all aggregated extractions.
    '''

    output_dict = {
        'files': [],
        'pca_path': ''
    }

    index_uuids = []
    for i, file_tup in enumerate(files_to_use):
        if file_tup[2]['uuid'] not in index_uuids:
            tmp = {
                'path': (file_tup[0], file_tup[1]),
                'uuid': file_tup[2]['uuid'],
                'group': 'default',
                'metadata': {'SessionName': f'default_{i}', 'SubjectName': f'default_{i}'} # fallback metadata
            }

            # handling metadata sub-dictionary values
            if 'metadata' in file_tup[2]:
                tmp['metadata'].update(file_tup[2]['metadata'])
            else:
                warnings.warn(f'Could not locate metadata for {file_tup[0]}! File will be listed with minimal default metadata.')
            
            index_uuids.append(file_tup[2]['uuid'])
            # appending file with default information
            output_dict['files'].append(tmp)

    return output_dict

# TODO: this doesn't look like it loads h5s - it looks like it loads metadata
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

        # Specific use case block: Behavior reinforcement experiments
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

    # Additional metadata for certain use cases
    additional_meta = []

    # Behavior reinforcement metadata
    additional_meta.append({
        'filename': 'feedback_ts.txt',
        'var_name': 'realtime_feedback',
        'dtype': np.bool,
    })

    # Pre-trained model real-time syllable classification results
    additional_meta.append({
        'filename': 'predictions.txt',
        'var_name': 'realtime_predictions',
        'dtype': np.int,
    })

    # Real-Time Recorded/Computed PC Scores
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
                    warnings.warn('WARNING: Did not load timestamps! This may cause issues if total dropped frames > 2% of the session.')

    return manifest

def copy_manifest_results(manifest, output_dir):
    '''
    Copies all consolidated manifest results to their respective output files.

    Parameters
    ----------
    manifest (dict): manifest dictionary containing all extraction h5 metadata to save
    output_dir (str): path to directory where extraction results will be aggregated.

    Returns
    -------
    None
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # now the key is the source h5 file and the value is the path to copy to
    for k, v in tqdm(manifest.items(), desc='Copying files'):

        if os.path.exists(os.path.join(output_dir, '{}.h5'.format(v['copy_path']))):
            continue

        basename = os.path.splitext(os.path.basename(k))[0]
        dirname = os.path.dirname(k)

        h5_path = k
        mp4_path = os.path.join(dirname, '{}.mp4'.format(basename))

        if os.path.exists(h5_path):
            new_h5_path = os.path.join(output_dir, '{}.h5'.format(v['copy_path']))
            shutil.copyfile(h5_path, new_h5_path)

        # if we have additional_meta then crack open the h5py and write to a safe place
        if len(v['additional_metadata']) > 0:
            for k2, v2 in v['additional_metadata'].items():
                new_key = '/metadata/misc/{}'.format(k2)
                with h5py.File(new_h5_path, "a") as f:
                    # TODO: stick with one string format throughout the repository.
                    # either f"{var}" or "{}".format(var).
                    f.create_dataset('{}/data'.format(new_key), data=v2["data"])
                    f.create_dataset('{}/timestamps'.format(new_key), data=v2["timestamps"])

        if os.path.exists(mp4_path):
            shutil.copyfile(mp4_path, os.path.join(
                output_dir, '{}.mp4'.format(v['copy_path'])))

        v['yaml_dict'].pop('extraction_metadata', None)
        with open('{}.yaml'.format(os.path.join(output_dir, v['copy_path'])), 'w') as f:
            yaml.safe_dump(v['yaml_dict'], f)

def handle_extract_metadata(input_file, dirname):
    '''
    Extracts metadata from input depth files, either raw or compressed.
    Locates metadata JSON file, and timestamps.txt file, then loads them into variables
    to be used to extract_wrapper.

    Parameters
    ----------
    input_file (str): path to input file to extract
    dirname (str): path to directory where extraction files reside.

    Returns
    -------
    input_file (str): path to decompressed input file (if input_file was originally a tarfile
    acquisition_metadata (dict): key-value pairs of JSON contents
    timestamps (1D array): list of loaded timestamps
    alternate_correct (bool): indicator for whether an alternate timestamp file was used
    tar (bool): indicator for whether the file is compressed.
    '''
    tar = None
    tar_members = None
    alternate_correct = False

    # Handle TAR files
    if input_file.endswith(('.tar.gz', '.tgz')):
        print(f'Scanning tarball {input_file} (this will take a minute)')
        # compute NEW psuedo-dirname now, `input_file` gets overwritten below with test_vid.dat tarinfo...
        dirname = os.path.join(dirname, os.path.basename(input_file).replace('.tar.gz', '').replace('.tgz', ''))

        tar = tarfile.open(input_file, 'r:gz')
        tar_members = tar.getmembers()
        tar_names = [_.name for _ in tar_members]
        # TODO: understand this - is this only here for tests? if so, it should be re-tooled
        input_file = tar_members[tar_names.index('test_vid.dat')]

    if tar is not None:
        # Handling tar paths
        metadata_path = tar.extractfile(tar_members[tar_names.index('metadata.json')])
        if "depth_ts.txt" in tar_names:
            timestamp_path = tar.extractfile(tar_members[tar_names.index('depth_ts.txt')])
        elif "timestamps.csv" in tar_names:
            timestamp_path = tar.extractfile(tar_members[tar_names.index('timestamps.csv')])
            alternate_correct = True
    else:
        # Handling non-compressed session paths
        metadata_path = os.path.join(dirname, 'metadata.json')
        timestamp_path = os.path.join(dirname, 'depth_ts.txt')
        alternate_timestamp_path = os.path.join(dirname, 'timestamps.csv')
        # Checks for alternative timestamp file if original .txt extension does not exist
        if not os.path.exists(timestamp_path) and os.path.exists(alternate_timestamp_path):
            timestamp_path = alternate_timestamp_path
            alternate_correct = True

    acquisition_metadata = load_metadata(metadata_path)
    timestamps = load_timestamps(timestamp_path, col=0, alternate=alternate_correct)

    return input_file, acquisition_metadata, timestamps, tar


# extract h5 helper function
def create_extract_h5(h5_file, acquisition_metadata, config_data, status_dict, scalars_attrs,
                      nframes, roi, bground_im, first_frame, timestamps, **kwargs):
    '''
    This is a helper function for extract_wrapper(); handles writing the following metadata
    to an open results_00.h5 file:
    Acquisition metadata, extraction metadata, computed scalars, timestamps, and original frames/frames_mask.

    Parameters
    ----------
    h5_file (h5py.File object): opened h5 file object to write to.
    acquisition_metadata (dict): Dictionary containing extracted session acquisition metadata.
    config_data (dict): dictionary object containing all required extraction parameters. (auto generated)
    status_dict (dict): dictionary that helps indicate if the session has been extracted fully.
    scalars_attrs (dict): dict of computed scalar attributes and descriptions to save.
    nframes (int): number of frames being recorded
    roi (2d np.ndarray): Computed 2D ROI Image.
    bground_im (2d np.ndarray): Computed 2D Background Image.
    first_frame (2d np.ndarray): Computed 2D First Frame Image.
    timestamps (np.array): Array of session timestamps.
    extract (moseq2_extract.cli.extract function): Used to preseve CLI state parameters in extraction h5.

    Returns
    -------
    None
    '''

    h5_file.create_dataset('metadata/uuid', data=status_dict['uuid'])

    # Creating scalar dataset
    for scalar in list(scalars_attrs.keys()):
        h5_file.create_dataset(f'scalars/{scalar}', (nframes,), 'float32', compression='gzip')
        h5_file[f'scalars/{scalar}'].attrs['description'] = scalars_attrs[scalar]

    # Timestamps
    if timestamps is not None:
        h5_file.create_dataset('timestamps', compression='gzip', data=timestamps)
        h5_file['timestamps'].attrs['description'] = "Depth video timestamps"

    # Cropped Frames
    h5_file.create_dataset('frames', (nframes, config_data['crop_size'][0], config_data['crop_size'][1]),
                     config_data['frame_dtype'], compression='gzip')
    h5_file['frames'].attrs['description'] = '3D Numpy array of depth frames (nframes x w x h, in mm)'

    # Frame Masks for EM Tracking
    if config_data['use_tracking_model']:
        h5_file.create_dataset('frames_mask', (nframes, config_data['crop_size'][0], config_data['crop_size'][1]), 'float32',
                         compression='gzip')
        h5_file['frames_mask'].attrs['description'] = 'Log-likelihood values from the tracking model (nframes x w x h)'
    else:
        h5_file.create_dataset('frames_mask', (nframes, config_data['crop_size'][0], config_data['crop_size'][1]), 'bool',
                         compression='gzip')
        h5_file['frames_mask'].attrs['description'] = 'Boolean mask, false=not mouse, true=mouse'

    # Flip Classifier
    if config_data['flip_classifier'] is not None:
        h5_file.create_dataset('metadata/extraction/flips', (nframes,), 'bool', compression='gzip')
        h5_file['metadata/extraction/flips'].attrs['description'] = 'Output from flip classifier, false=no flip, true=flip'

    # True Depth
    h5_file.create_dataset('metadata/extraction/true_depth', data=config_data['true_depth'])
    h5_file['metadata/extraction/true_depth'].attrs['description'] = 'Detected true depth of arena floor in mm'

    # ROI
    h5_file.create_dataset('metadata/extraction/roi', data=roi, compression='gzip')
    h5_file['metadata/extraction/roi'].attrs['description'] = 'ROI mask'

    # First Frame
    h5_file.create_dataset('metadata/extraction/first_frame', data=first_frame[0], compression='gzip')
    h5_file['metadata/extraction/first_frame'].attrs['description'] = 'First frame of depth dataset'

    # Background
    h5_file.create_dataset('metadata/extraction/background', data=bground_im, compression='gzip')
    h5_file['metadata/extraction/background'].attrs['description'] = 'Computed background image'

    # Extract Version
    extract_version = np.string_(get_distribution('moseq2-extract').version)
    h5_file.create_dataset('metadata/extraction/extract_version', data=extract_version)
    h5_file['metadata/extraction/extract_version'].attrs['description'] = 'Version of moseq2-extract'

    # Extraction Parameters
    from moseq2_extract.cli import extract
    dict_to_h5(h5_file, status_dict['parameters'], 'metadata/extraction/parameters', click_param_annot(extract))

    # Acquisition Metadata
    for key, value in acquisition_metadata.items():
        if type(value) is list and len(value) > 0 and type(value[0]) is str:
            value = [n.encode('utf8') for n in value]

        if value is not None:
            h5_file.create_dataset(f'metadata/acquisition/{key}', data=value)
        else:
            h5_file.create_dataset(f'metadata/acquisition/{key}', dtype="f")