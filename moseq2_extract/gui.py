import json
import warnings
from .cli import *
from glob import glob
from pathlib import Path
import ruamel.yaml as yaml
from cytoolz import partial
from moseq2_extract.helpers.data import get_selected_sessions, load_h5s, build_manifest, copy_manifest_results
from moseq2_extract.helpers.extract import run_local_extract, run_slurm_extract
from moseq2_extract.helpers.wrappers import get_roi_wrapper, extract_wrapper, flip_file_wrapper, generate_index_wrapper
from moseq2_extract.io.image import read_image
from moseq2_extract.util import (recursive_find_h5s, escape_path,
                                 mouse_threshold_filter, recursive_find_unextracted_dirs)

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

    with open(progress_file, 'r') as f:
        progress = yaml.safe_load(f)

    progress[varK] = varV
    with open(progress_file, 'w') as f:
        yaml.safe_dump(progress, f)

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
    f.close()

    return vars['config_file'], vars['index_file'], vars['train_data_dir'], vars['pca_dirname'], vars['scores_filename'], vars['model_path'], vars['scores_path'], vars['crowd_dir'], vars['plot_path']

def check_progress(base_dir, progress_filepath, output_directory=None):
    '''
    Checks whether progress file exists and prompts user input on whether to overwrite, load old, or generate a new one.

    Parameters
    ----------
    base_dir (str): path to directory to create/find progress file
    progress_filepath (str): path to progress filename
    output_directory (str): optional alternative output directory path.

    Returns
    -------
    All restored variables or None.
    '''

    if output_directory is not None:
        progress_filepath = os.path.join(output_directory, progress_filepath.split('/')[-1])

    if os.path.exists(progress_filepath):
        with open(progress_filepath, 'r') as f:
            progress_vars = yaml.safe_load(f)
        f.close()

        print('Progress file found, listing initialized variables...\n')

        for k, v in progress_vars.items():
            if v != 'TBD':
                print(f'{k}: {v}')

        restore = ''
        while(restore != 'Y' or restore != 'N' or restore != 'q'):
            restore = input('Would you like to restore the above listed notebook variables? [Y -> restore variables, N -> overwrite progress file, q -> quit]')
            if restore == "Y":

                print('Updating notebook variables...')

                config_filepath, index_filepath, train_data_dir, pca_dirname, \
                scores_filename, model_path, scores_file, \
                crowd_dir, plot_path = restore_progress_vars(progress_filepath)

                return config_filepath, index_filepath, train_data_dir, pca_dirname, \
                scores_filename, model_path, scores_file, \
                crowd_dir, plot_path
            elif restore == "N":

                print('Overwriting progress file.')

                progress_vars = {'base_dir': base_dir, 'config_file': 'TBD', 'index_file': 'TBD', 'train_data_dir': 'TBD',
                                 'pca_dirname': 'TBD', 'scores_filename': 'TBD', 'scores_path': 'TBD', 'model_path': 'TBD',
                                 'crowd_dir': 'TBD', 'plot_path': 'TBD'}

                with open(progress_filepath, 'w') as f:
                    yaml.safe_dump(progress_vars, f)
                f.close()

                print('\nProgress file created, listing initialized variables...')
                for k, v in progress_vars.items():
                    if v != 'TBD':
                        print(k, v)
                return progress_vars['config_file'], progress_vars['index_file'], progress_vars['train_data_dir'], progress_vars['pca_dirname'], progress_vars['scores_filename'], \
                       progress_vars['model_path'], progress_vars['scores_path'], progress_vars['crowd_dir'], progress_vars['plot_path']
            elif restore == 'q':
                return
    else:
        print('Progress file not found, creating new one.')
        progress_vars = {'base_dir': base_dir, 'config_file': 'TBD', 'index_file': 'TBD', 'train_data_dir': 'TBD', 'pca_dirname': 'TBD',
                         'scores_filename': 'TBD', 'scores_path': 'TBD', 'model_path': 'TBD', 'crowd_dir': 'TBD', 'plot_path': 'TBD'}

        with open(progress_filepath, 'w') as f:
            yaml.safe_dump(progress_vars, f)
        f.close()

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
    if os.path.exists(output_file):
        print('This file already exists, would you like to overwrite it? [Y -> yes, else -> exit]')
        ow = input()
        if ow == 'Y':
            with open(output_file, 'w') as f:
                yaml.safe_dump(params, f)
        else:
            return 'Configuration file has been retained'
    else:
        print('Creating configuration file.')
        with open(output_file, 'w') as f:
            yaml.safe_dump(params, f)

    return 'Configuration file has been successfully generated.'

def view_extraction(extractions):
    '''
    Prompts user to select which extracted video(s) to preview.

    Parameters
    ----------
    extractions (list): list of paths to all extracted avi videos.

    Returns
    -------
    extractions (list): list of selected extractions.
    '''

    if len(extractions) > 1:
        for i, sess in enumerate(extractions):
            print(f'[{str(i + 1)}] {sess}')
    else:
        print('no sessions to view')

    while (True):
        try:
            input_file_indices = input(
                "Input extracted session indices to view separated by commas, or empty string for all sessions.\n").strip()
            if ',' in input_file_indices:
                input_file_indices = input_file_indices.split(',')
                for i in input_file_indices:
                    i = int(i.strip())
                    if i > len(extractions):
                        print('invalid index try again.')
                        input_file_index = []
                        break

                tmp = []
                for index in input_file_indices:
                    index = int(index.strip())
                    tmp.append(extractions[index - 1])
                extractions = tmp
                break
            elif len(input_file_indices.strip()) == 1:
                index = int(input_file_indices.strip())
                extractions = [extractions[index - 1]]
                break
            elif input_file_indices == '':
                break
        except:
            print('unexpected error:', sys.exc_info()[0])

    return extractions

def extract_found_sessions(input_dir, config_file, ext, extract_all=True, skip_extracted=False, output_directory=None):
    '''
    Searches for all depth files within input_directory with selected extension

    Parameters
    ----------
    input_dir (str): path to directory containing all session folders
    config_file (str): path to config file
    ext (str): file extension to search for
    extract_all (bool): if True, auto searches for all sessions, else, prompts user to select sessions individually.
    skip_extracted (bool): indicates whether to skip already extracted session.
    output_directory (str): optional alternative output_directory.

    Returns
    -------
    None
    '''

    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # find directories with .dat files that either have incomplete or no extractions
    partition = 'short'
    skip_checks = True
    config_file = Path(config_file)

    prefix = ''
    to_extract = recursive_find_unextracted_dirs(input_dir, filename=ext, skip_checks=skip_checks)
    to_extract = [e for e in to_extract if e.endswith(ext)]
    temp = []
    for dir in to_extract:
        if '/tmp/' not in dir:
            temp.append(dir)

    to_extract = temp

    to_extract = get_selected_sessions(to_extract, extract_all)

    if config_file is None:
        raise RuntimeError('Need a config file to continue')
    elif not os.path.exists(config_file):
        raise IOError(f'Config file {config_file} does not exist')

    with config_file.open() as f:
        params = yaml.safe_load(f)

    cluster_type = params.get('cluster_type', 'local')

    if type(params['bg_roi_index']) is int:
        params['bg_roi_index'] = [params['bg_roi_index']]

    if cluster_type == 'slurm':
        run_slurm_extract(to_extract, params, partition, prefix, escape_path, skip_extracted, output_directory)

    elif cluster_type == 'local':
        run_local_extract(to_extract, params, prefix, skip_extracted, output_directory)

    else:
        raise NotImplementedError('Other cluster types not supported')

    print('Extractions Complete.')

def generate_index_command(input_dir, pca_file, output_file, filter, all_uuids):
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

    output_file = generate_index_wrapper(input_dir, pca_file, output_file, filter, all_uuids)
    print('Index file successfully generated.')
    return output_file


def aggregate_extract_results_command(input_dir, format, output_dir, output_directory=None):
    '''
    Finds all extracted h5, yaml and avi files and copies them all to a
    new directory relabeled with their respective session names.
    Also generates the index file.

    Parameters
    ----------
    input_dir (str): path to base directory to recursively search for h5s
    format (str): filename format for info to include in filenames
    output_dir (str): path to directory to save all aggregated results
    output_directory (str): alternate path to save results

    Returns
    -------
    indexpath (str): path to newly generated index file.
    '''

    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    mouse_threshold = 0
    snake_case = True
    if output_directory is None:
        output_dir = os.path.join(input_dir, output_dir)
    else:
        output_dir = os.path.join(output_directory, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    h5s, dicts, _ = recursive_find_h5s(input_dir)

    not_in_output = lambda f: not os.path.exists(os.path.join(output_dir, os.path.basename(f)))
    complete = lambda d: d['complete'] and not d['skip']

    # only include real extracted mice with this filter func
    mtf = partial(mouse_threshold_filter, thresh=mouse_threshold)

    def filter_h5(args):
        '''remove h5's that should be skipped or extraction wasn't complete'''
        _dict, _h5 = args
        return complete(_dict) and not_in_output(_h5) and mtf(_h5)

    # load in all of the h5 files, grab the extraction metadata, reformat to make nice 'n pretty
    # then stage the copy
    to_load = list(filter(filter_h5, zip(dicts, h5s)))

    loaded = load_h5s(to_load)

    manifest = build_manifest(loaded, format=format)

    copy_manifest_results(manifest, output_dir)

    print('Results successfully aggregated in', output_dir)

    if output_directory is None:
        indexpath = generate_index_command(output_dir, '', os.path.join(input_dir, 'moseq2-index.yaml'), (), False)
    else:
        indexpath = generate_index_command(input_dir, '', os.path.join(output_directory, 'moseq2-index.yaml'), (), False)

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

    found_sessions = 0
    sessions = []
    for ext in exts:
        if len(data_dir) == 0:
            data_dir = os.getcwd()
            files = sorted(glob('*/*.'+ext))
            found_sessions += len(files)
            sessions += files
        else:
            data_dir = data_dir.strip()
            if os.path.isdir(data_dir):
                files = sorted(glob(os.path.join(data_dir,'*/*.'+ext)))
                found_sessions += len(files)
                sessions += files
            else:
                print('directory not found, try again.')

    # generate sample metadata json for each session that is missing one
    sample_meta = {'SubjectName': 'default', 'SessionName': 'default',
                   'NidaqChannels': 0, 'NidaqSamplingRate': 0.0, 'DepthResolution': [512, 424],
                   'ColorDataType': "Byte[]", "StartTime": ""}
    for sess in sessions:
        sess_dir = '/'.join(sess.split('/')[:-1])
        sess_name = sess.split('/')[-2]
        if 'metadata.json' not in os.listdir(sess_dir):
            sample_meta['SessionName'] = sess_name
            with open(os.path.join(sess_dir, 'metadata.json'), 'w') as fp:
                json.dump(sample_meta, fp)

    for i, sess in enumerate(sessions):
        print(f'[{str(i+1)}] {sess}')

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

    flip_file_wrapper(config_file, output_dir, selected_flip=selection, gui=True)


def find_roi_command(input_dir, config_file, exts=['dat', 'mkv', 'avi'], output_directory=None):
    '''
    Computes ROI files given depth file

    Parameters
    ----------
    input_dir (str): path to directory containing depth file
    config_file (str): path to config file
    exts (list): list of supported extensions
    output_directory (str): alternate output path

    Returns
    -------
    images (list of 2d arrays): list of 2d array images to graph in Notebook.
    filenames (list): list of paths to respective image paths
    '''

    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    files = []
    for ext in exts:
        files += sorted(glob(os.path.join(input_dir, '*/*.'+ext)))

    for i, sess in enumerate(files):
        print(f'[{str(i+1)}] {sess}')

    if len(files) == 0:
        print('No recordings found')
        return

    input_file_index = -1
    while(int(input_file_index) < 0):
        try:
            input_file_index = int(input("Input session index to find rois: ").strip())
            if int(input_file_index) > len(files):
                print('invalid index try again.')
                input_file_index = -1
        except:
            print('invalid input, only input integers. Input Q to exit.')
            if 'q' in str(input_file_index).lower():
                return

    input_file = files[input_file_index - 1]

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    output_dir = get_roi_wrapper(input_file, config_data, output_directory, gui=True)

    images = []
    filenames = []
    for infile in os.listdir(output_dir):
        if infile[-4:] == "tiff":
            im = read_image(os.path.join(output_dir, infile))
            if len(im.shape) == 2:
                images.append(im)
            elif len(im.shape) == 3:
                images.append(im[0])
            filenames.append(infile)

    print(f'ROIs were successfully computed in {output_dir}')
    return images, filenames

def sample_extract_command(input_dir, config_file, nframes, output_directory=None, exts=['dat', 'mkv', 'avi']):
    '''
    Test extract command to extract a subset of the video.

    Parameters
    ----------
    input_dir (str): path to directory containing depth file to extract
    config_file (str): path to config file
    nframes (int): number of frames to extract
    output_directory (str): path to alternative directory
    exts (list): list of supported depth file extensions.

    Returns
    -------
    output_dir (str): path to directory containing sample extraction results.
    '''

    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    files = []
    for ext in exts:
        files += sorted(glob(os.path.join(input_dir, '*/*.'+ext.replace('.',''))))

    files = sorted(files)

    if len(files) == 0:
        print('No recordings found')
        return

    for i, sess in enumerate(files):
        print(f'[{str(i + 1)}] {sess}')
    input_file_index = -1

    while (int(input_file_index) < 0):
        try:
            input_file_index = int(input("Input session index to extract sample: ").strip())
            if int(input_file_index) > len(files):
                print('invalid index try again.')
                input_file_index = -1
        except:
            print('invalid input, only input integers.')

    if output_directory is None:
        output_dir = os.path.join(os.path.dirname(files[input_file_index-1]), 'sample_proc')
    else:
        output_dir = os.path.join(output_directory, 'sample_proc')

    input_file = files[input_file_index - 1]

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