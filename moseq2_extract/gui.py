'''
GUI front-end operations accessible from a jupyter notebook.

This module contains all operations included in the CLI module,
with some additional preprocessing steps and state-retrieval functionality
to facilitate Jupyter notebook usage.
'''

import os
import ruamel.yaml as yaml
from ast import literal_eval
from os.path import dirname, basename, exists, join
from moseq2_extract.util import read_yaml
from moseq2_extract.io.image import read_tiff_files
from moseq2_extract.helpers.extract import run_local_extract
from moseq2_extract.helpers.wrappers import get_roi_wrapper, extract_wrapper, flip_file_wrapper, \
                                            generate_index_wrapper, aggregate_extract_results_wrapper
from moseq2_extract.util import (recursive_find_unextracted_dirs, load_found_session_paths, filter_warnings)

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
            ss = s.strip('e ')
            if '-' not in ss:
                if isinstance(literal_eval(ss), int):
                    excluded_sess_idx.append(int(ss))
            else:
                ssd = ss.split('-')
                if isinstance(literal_eval(ssd[0]), int) and isinstance(literal_eval(ssd[1]), int):
                    for i in range(int(ssd[0]), int(ssd[1]) + 1):
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
                if len(selected_sess_idx) > 0:
                    iters = selected_sess_idx
                else:
                    iters = range(1, len(to_extract)+1)
                for i in iters:
                    if i not in excluded_sess_idx:
                        if i-1 < len(to_extract):
                            ret_extract.append(to_extract[i - 1])
            else:
                print('Invalid input. Try again or press q to quit.')
    else:
        return to_extract

    return ret_extract

@filter_warnings
def generate_config_command(output_file):
    '''
    Generates configuration file to use throughout pipeline.

    Parameters
    ----------
    output_file (str): path to saved config file.

    Returns
    -------
    (str): status message.
    '''

    from .cli import extract
    objs = extract.params

    params = {tmp.name: tmp.default for tmp in objs if not tmp.required}

    # Check if the file already exists, and prompt user if they would like to overwrite pre-existing file
    if exists(output_file):
        ow = input('This file already exists, would you like to overwrite it? [y -> yes, n -> no] ')
        if ow.lower() == 'y':
            # Updating config file
            with open(output_file, 'w') as f:
                yaml.safe_dump(params, f)
        else:
            return 'Configuration file has been retained'
    else:
        print('Creating configuration file.')
        with open(output_file, 'w') as f:
            yaml.safe_dump(params, f)

    return 'Configuration file has been successfully generated.'


@filter_warnings
def extract_found_sessions(input_dir, config_file, ext, extract_all=True, skip_extracted=False):
    '''
    Searches for all depth files within input_directory with selected extension

    Parameters
    ----------
    input_dir (str): path to directory containing all session folders
    config_file (str): path to config file
    ext (str): file extension to search for
    extract_all (bool): if True, auto searches for all sessions, else, prompts user to select sessions individually.
    skip_extracted (bool): indicates whether to skip already extracted session.

    Returns
    -------
    None
    '''
    # error out early
    if not exists(config_file):
        raise IOError(f'Config file {config_file} does not exist')

    to_extract = []

    # find directories with .dat files that either have incomplete or no extractions
    if isinstance(ext, str):
        ext = [ext]
    for ex in ext:
        tmp = recursive_find_unextracted_dirs(input_dir, extension=ex, skip_checks=True)
        to_extract += [e for e in tmp if e.endswith(ex)]

    # filter out any incorrectly returned sessions
    temp = sorted([sess_dir for sess_dir in to_extract if '/tmp/' not in sess_dir])
    to_extract = get_selected_sessions(temp, extract_all)

    run_local_extract(to_extract, config_file, skip_extracted)

    print('Extractions Complete.')

def generate_index_command(input_dir, output_file):
    '''
    Generates Index File based on aggregated sessions

    Parameters
    ----------
    input_dir (str): path to aggregated_results/ dir
    output_file (str): index file name

    Returns
    -------
    output_file (str): path to index file.
    '''

    output_file = generate_index_wrapper(input_dir, output_file)
    print('Index file successfully generated.')
    return output_file


@filter_warnings
def aggregate_extract_results_command(input_dir, format, output_dir, mouse_threshold=0.0):
    '''
    Finds all extracted h5, yaml and avi files and copies them all to a
    new directory relabeled with their respective session names.
    Also generates the index file.

    Parameters
    ----------
    input_dir (str): path to base directory to recursively search for h5s
    format (str): filename format for info to include in filenames
    output_dir (str): path to directory to save all aggregated results
    mouse_threshold (float): threshold value of mean frame depth to include session frames

    Returns
    -------
    indexpath (str): path to newly generated index file.
    '''

    output_dir = join(input_dir, output_dir)

    if not exists(output_dir):
        os.makedirs(output_dir)

    indexpath = aggregate_extract_results_wrapper(input_dir, format, output_dir, mouse_threshold)

    return indexpath

def download_flip_command(output_dir, config_file="", selection=1):
    '''
    Downloads flip classifier and saves its path in the inputted config file

    Parameters
    ----------
    output_dir (str): path to output directory to save flip classifier
    config_file (str): path to config file
    selection (int): index of which flip file to download (default is Adult male C57 classifer)

    Returns
    -------
    None
    '''

    flip_file_wrapper(config_file, output_dir, selected_flip=selection)


@filter_warnings
def find_roi_command(input_dir, config_file, exts=['dat', 'mkv', 'avi'], select_session=False, default_session=0):
    '''
    Computes ROI files given depth file.
    Will list out all available sessions to process and prompts user to input a corresponding session
    index to process.

    Parameters
    ----------
    input_dir (str): path to directory containing depth file
    config_file (str): path to config file
    exts (list): list of supported extensions
    select_session (bool): list all found sessions and allow user to select specific session to analyze via user-prompt
    default_session (int): index of the default session to find ROI for

    Returns
    -------
    images (list of 2d arrays): list of 2d array images to graph in Notebook.
    filenames (list): list of paths to respective image paths
    '''

    files = load_found_session_paths(input_dir, exts)

    if len(files) == 0:
        print('No recordings found')
        return

    if select_session:
        input_file = get_selected_sessions(files, False)
        if isinstance(input_file, list):
            input_file = input_file[0]
    else:
        input_file = files[default_session]

    print(f'Processing session: {input_file}')
    config_data = read_yaml(config_file)

    output_dir = join(dirname(input_file), 'proc')
    get_roi_wrapper(input_file, config_data, output_dir)

    with open(config_file, 'w') as g:
        yaml.safe_dump(config_data, g)

    images, filenames = read_tiff_files(output_dir)

    print(f'ROIs were successfully computed in {output_dir}')
    return images, filenames


@filter_warnings
def extract_command(input_file, output_dir, config_file, num_frames=None, skip=False):
    '''
    Command to extract a full depth file

    Parameters
    ----------
    input_file (str): path to depthfile
    output_dir (str): path to output directory
    config_file (str): path to config file
    num_frames (int): number of frames to extract. All if None.
    skip (bool): skip already extracted file.

    Returns
    -------
    None
    '''

    config_data = read_yaml(config_file)

    # Loading individual session config parameters if it exists
    if exists(config_data.get('session_config_path', '')):
        session_configs = read_yaml(config_data['session_config_path'])
        session_key = basename(dirname(input_file))

        # If key is found, update config_data, otherwise, use default dict
        config_data = session_configs.get(session_key, config_data)

    if output_dir is None:
        output_dir = config_data.get('output_dir', 'proc')

    extract_wrapper(input_file, output_dir, config_data, num_frames=num_frames, skip=skip)

    return 'Extraction completed.'