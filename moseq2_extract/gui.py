'''
GUI front-end operations accessible from a jupyter notebook.

This module contains all operations included in the CLI module,
with some additional preprocessing steps and state-retrieval functionality
to facilitate Jupyter notebook usage.
'''

import json
import warnings
from .cli import *
import ruamel.yaml as yaml
from moseq2_extract.io.image import read_tiff_files
from moseq2_extract.helpers.extract import run_local_extract
from moseq2_extract.helpers.data import get_selected_sessions
from moseq2_extract.util import recursive_find_unextracted_dirs, load_found_session_paths
from moseq2_extract.helpers.wrappers import get_roi_wrapper, extract_wrapper, flip_file_wrapper, \
                                            generate_index_wrapper, aggregate_extract_results_wrapper

def update_progress(progress_file, varK, varV):
    '''
    Updates progress file with new notebook variable

    Parameters
    ----------
    progress_file (str): path to progress file
    varK (str): key in progress file to update
    varV (str): updated value to write

    Returns
    -------
    None
    '''

    yml = yaml.YAML()
    yml.indent(mapping=2, offset=2)

    with open(progress_file, 'r') as f:
        progress = yaml.safe_load(f)

    progress[varK] = varV
    with open(progress_file, 'w') as f:
        yml.dump(progress, f)

    print(f'Successfully updated progress file with {varK} -> {varV}')

def restore_progress_vars(progress_file):
    '''
    Restore all saved progress variables to Jupyter Notebook.

    Parameters
    ----------
    progress_file (str): path to progress file

    Returns
    -------
    All progress file variables
    '''

    with open(progress_file, 'r') as f:
        vars = yaml.safe_load(f)

    return vars['config_file'], vars['index_file'], vars['train_data_dir'], vars['pca_dirname'], vars['scores_filename'], vars['model_path'], vars['scores_path'], vars['crowd_dir'], vars['plot_path']

def check_progress(base_dir, progress_filepath):
    '''
    Checks whether progress file exists and prompts user input on whether to overwrite, load old, or generate a new one.

    Parameters
    ----------
    base_dir (str): path to directory to create/find progress file
    progress_filepath (str): path to progress filename

    Returns
    -------
    All restored variables or None.
    '''

    yml = yaml.YAML()
    yml.indent(mapping=2, offset=2)

    base_progress_vars = {'base_dir': base_dir, 'config_file': 'TBD', 'index_file': 'TBD', 'train_data_dir': 'TBD',
                     'pca_dirname': 'TBD', 'scores_filename': 'TBD', 'scores_path': 'TBD', 'model_path': 'TBD',
                     'crowd_dir': 'TBD', 'plot_path': os.path.join(base_dir, 'plots/')}

    if os.path.exists(progress_filepath):
        with open(progress_filepath, 'r') as f:
            progress_vars = yaml.safe_load(f)

        print('Progress file found, listing initialized variables...\n')

        for k, v in progress_vars.items():
            if v != 'TBD':
                print(f'{k}: {v}')

        restore = ''
        # Restore loaded variables or overwrite with fresh state
        while(restore != 'Y' or restore != 'N' or restore != 'q'):
            restore = input('Would you like to restore the above listed notebook variables? [Y -> restore variables, N -> overwrite progress file, q -> quit]')
            if restore.lower() == "y":

                print('Updating notebook variables...')

                config_filepath, index_filepath, train_data_dir, pca_dirname, \
                scores_filename, model_path, scores_file, \
                crowd_dir, plot_path = restore_progress_vars(progress_filepath)

                return config_filepath, index_filepath, train_data_dir, pca_dirname, \
                scores_filename, model_path, scores_file, crowd_dir, plot_path

            elif restore.lower() == "n":

                print('Overwriting progress file with initial progress.')

                progress_vars = base_progress_vars

                with open(progress_filepath, 'w') as f:
                    yaml.safe_dump(progress_vars, f)

                print('\nProgress file created, listing initialized variables...')
                for k, v in progress_vars.items():
                    if v != 'TBD':
                        print(k, v)
                return progress_vars['config_file'], progress_vars['index_file'], progress_vars['train_data_dir'], progress_vars['pca_dirname'], progress_vars['scores_filename'], \
                       progress_vars['model_path'], progress_vars['scores_path'], progress_vars['crowd_dir'], progress_vars['plot_path']
            elif restore.lower() == 'q':
                return
    else:
        print('Progress file not found, creating new one.')
        progress_vars = base_progress_vars

        with open(progress_filepath, 'w') as f:
            yml.dump(progress_vars, f)

        print('\nProgress file created, listing initialized variables...')
        for k, v in progress_vars.items():
            if v != 'TBD':
                print(k, v)

        return progress_vars['config_file'], progress_vars['index_file'], progress_vars['train_data_dir'], progress_vars['pca_dirname'], progress_vars['scores_filename'],\
               progress_vars['model_path'], progress_vars['scores_path'], progress_vars['crowd_dir'], progress_vars['plot_path']

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

    warnings.simplefilter(action='ignore', category=yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    objs = extract.params

    params = {tmp.name: tmp.default for tmp in objs if not tmp.required}

    input_dir = os.path.dirname(output_file)
    params['input_dir'] = input_dir
    params['detected_true_depth'] = 'auto'

    # Check if the file already exists, and prompt user if they would like to overwrite pre-existing file
    if os.path.exists(output_file):
        print('This file already exists, would you like to overwrite it? [Y -> yes, else -> exit]')
        ow = input()
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

def view_extraction(extractions, default=0):
    '''
    Prompts user to select which extracted video(s) to preview.

    Parameters
    ----------
    extractions (list): list of paths to all extracted avi videos.
    default (int): index of the default extraction to display

    Returns
    -------
    extractions (list): list of selected extractions.
    '''

    if len(extractions) == 0:
        print('no sessions to view')
        return []

    if default < 0:
        for i, sess in enumerate(extractions):
            print(f'[{str(i + 1)}] {sess}')
        extractions = get_selected_sessions(extractions, False)
    else:
        print(f"Displaying {extractions[default]}")
        return [extractions[default]]

    return extractions

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

    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    to_extract = []

    # find directories with .dat files that either have incomplete or no extractions
    if isinstance(ext, str):
        to_extract = recursive_find_unextracted_dirs(input_dir, filename=ext, skip_checks=True)
        to_extract = [e for e in to_extract if e.endswith(ext)]
    elif isinstance(ext, list):
        for ex in ext:
            tmp = recursive_find_unextracted_dirs(input_dir, filename=ex, skip_checks=True)
            to_extract += [e for e in tmp if e.endswith(ex)]

    # filter out any incorrectly returned sessions
    temp = [sess_dir for sess_dir in to_extract if '/tmp/' not in sess_dir]
    to_extract = get_selected_sessions(temp, extract_all)

    if not os.path.exists(config_file):
        raise IOError(f'Config file {config_file} does not exist')

    run_local_extract(to_extract, config_file, skip_extracted)

    print('Extractions Complete.')

def generate_index_command(input_dir, pca_file, output_file, filter, all_uuids, subpath='/'):
    '''
    Generates Index File based on aggregated sessions

    Parameters
    ----------
    input_dir (str): path to aggregated_results/ dir
    pca_file (str): path to pca file
    output_file (str): index file name
    filter (list): keys to filter through
    all_uuids (list): all extracted session uuids

    Returns
    -------
    output_file (str): path to index file.
    '''

    output_file = generate_index_wrapper(input_dir, pca_file, output_file, filter, all_uuids, subpath=subpath)
    print('Index file successfully generated.')
    return output_file

def aggregate_extract_results_command(input_dir, format, output_dir, subpath='/'):
    '''
    Finds all extracted h5, yaml and avi files and copies them all to a
    new directory relabeled with their respective session names.
    Also generates the index file.

    Parameters
    ----------
    input_dir (str): path to base directory to recursively search for h5s
    format (str): filename format for info to include in filenames
    output_dir (str): path to directory to save all aggregated results

    Returns
    -------
    indexpath (str): path to newly generated index file.
    '''

    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    output_dir = os.path.join(input_dir, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    aggregate_extract_results_wrapper(input_dir, format, output_dir)

    indexpath = generate_index_command(output_dir, '', os.path.join(input_dir, 'moseq2-index.yaml'), (), False, subpath=subpath)

    print(f'Index file path: {indexpath}')

    return indexpath

def get_found_sessions(data_dir="", exts=['dat', 'mkv', 'avi']):
    '''
    Find all depth recording sessions (with given extensions) to work on given base directory.

    Parameters
    ----------
    data_dir (str): path to directory containing all session folders
    exts (list): list of depth file extensions to search for

    Returns
    -------
    data_dir (str): path to base_dir to save in progress file
    found_sessions (int): number of found sessions with given extensions
    '''

    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    sessions = load_found_session_paths(data_dir, exts)

    # generate sample metadata json for each session that is missing one
    sample_meta = {'SubjectName': 'default', 'SessionName': 'default',
                   'NidaqChannels': 0, 'NidaqSamplingRate': 0.0, 'DepthResolution': [512, 424],
                   'ColorDataType': "Byte[]", "StartTime": ""}

    for sess in sessions:
        sess_dir = '/'.join(sess.split('/')[:-1]) # get path to session directory
        sess_name = sess.split('/')[-2]
        if 'metadata.json' not in os.listdir(sess_dir):
            warnings.warn(f'No metadata file corresponding to {sess_name} found. Generating a default template.')
            sample_meta['SessionName'] = sess_name

            # Generating a default metadata.json in respective session directory
            with open(os.path.join(sess_dir, 'metadata.json'), 'w') as fp:
                json.dump(sample_meta, fp)

    for i, sess in enumerate(sessions):
        print(f'[{str(i+1)}] {sess}')

    found_sessions = len(sessions)

    return data_dir, found_sessions

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

    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

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
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    output_dir = get_roi_wrapper(input_file, config_data, gui=True)

    with open(config_file, 'w') as g:
        yaml.safe_dump(config_data, g)

    images, filenames = read_tiff_files(output_dir)

    print(f'ROIs were successfully computed in {output_dir}')
    return images, filenames

def sample_extract_command(input_dir, config_file, nframes, select_session=False, default_session=0, exts=['dat', 'mkv', 'avi']):
    '''
    Test extract command to extract a subset of the video.

    Parameters
    ----------
    input_dir (str): path to directory containing depth file to extract
    config_file (str): path to config file
    nframes (int): number of frames to extract
    select_session (bool): list all found sessions and allow user to select specific session to analyze via user-prompt
    default_session (int): index of the default session to find ROI for
    exts (list): list of supported depth file extensions.

    Returns
    -------
    output_dir (str): path to directory containing sample extraction results.
    '''

    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

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

    output_dir = os.path.join(os.path.dirname(input_file), 'sample_proc')

    extract_command(input_file, output_dir, config_file, num_frames=nframes)
    print(f'Sample extraction of {nframes} frames completed successfully in {output_dir}.')

    return output_dir

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

    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    extract_wrapper(input_file, output_dir, config_data, num_frames=num_frames, skip=skip, gui=True)

    return 'Extraction completed.'