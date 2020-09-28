'''
Wrapper functions for all functionality afforded by MoSeq2-Extract.
These functions perform all the data processing from start to finish, and are shared between the CLI and GUI.
'''

import os
import uuid
import h5py
import shutil
import warnings
import numpy as np
import urllib.request
from copy import deepcopy
import ruamel.yaml as yaml
from tqdm.auto import tqdm
from cytoolz import partial
from moseq2_extract.io.image import write_image
from moseq2_extract.util import mouse_threshold_filter
from moseq2_extract.helpers.extract import process_extract_batches
from moseq2_extract.io.video import load_movie_data, get_movie_info
from moseq2_extract.extract.proc import get_roi, get_bground_im_file
from moseq2_extract.helpers.data import handle_extract_metadata, create_extract_h5, load_h5s, build_manifest, \
                            copy_manifest_results, build_index_dict, check_completion_status
from moseq2_extract.util import select_strel, gen_batch_sequence, scalar_attributes, convert_raw_to_avi_function, \
                        set_bground_to_plane_fit, recursive_find_h5s, clean_dict, graduate_dilated_wall_area, \
                        h5_to_dict, set_bg_roi_weights, get_frame_range_indices, check_filter_sizes, get_strels

def copy_h5_metadata_to_yaml_wrapper(input_dir, h5_metadata_path):
    '''
    Copy's user specified metadata from h5path to a yaml file.

    Parameters
    ----------
    input_dir (str): path to directory containing h5 files
    h5_metadata_path (str): path within h5 to desired metadata to copy to yaml.

    Returns
    -------
    None
    '''

    h5s, dicts, yamls = recursive_find_h5s(input_dir)
    to_load = [(tmp, yml, file) for tmp, yml, file in zip(
        dicts, yamls, h5s) if tmp['complete'] and not tmp['skip']]

    # load in all of the h5 files, grab the extraction metadata, reformat to make nice 'n pretty
    # then stage the copy

    for i, tup in tqdm(enumerate(to_load), total=len(to_load), desc='Copying data to yamls'):
        with h5py.File(tup[2], 'r') as f:
            tmp = clean_dict(h5_to_dict(f, h5_metadata_path))
            tup[0]['metadata'] = dict(tmp)

        try:
            new_file = '{}_update.yaml'.format(os.path.basename(tup[1]))
            with open(new_file, 'w+') as f:
                yaml.safe_dump(tup[0], f)
            shutil.move(new_file, tup[1])
        except Exception:
            raise Exception


def generate_index_wrapper(input_dir, output_file, subpath='proc/'):
    '''
    Generates index file containing a summary of all extracted sessions.

    Parameters
    ----------
    input_dir (str): directory to search for extracted sessions.
    output_file (str): preferred name of the index file.
    subpath (str): subdirectory that all sessions must exist within

    Returns
    -------
    output_file (str): path to index file.
    '''

    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # gather the h5s and the pca scores file
    # uuids should match keys in the scores file
    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    file_with_uuids = [(os.path.abspath(h5), os.path.abspath(yml), meta) for h5, yml, meta in zip(h5s, yamls, dicts)]

    # Ensuring all retrieved extracted session h5s have the appropriate metadata
    # included in their results_00.h5 file
    try:
        for file in file_with_uuids:
            if 'metadata' not in file[2]:
                copy_h5_metadata_to_yaml_wrapper(input_dir, file[0])
    except:
        warnings.warn(f'Metadata for session {file[0]} not found. \
        File may be listed with minimal/defaulted metadata in index file.')

    # Filtering out sessions that do not contain the required subpath in their paths.
    # Ensures that there are no sample extractions included in the index file.
    keep_samples = [i for i, f in enumerate(file_with_uuids) if subpath in f[0]]
    files_to_use = [tup for i, tup in enumerate(file_with_uuids) if i in keep_samples]

    print(f'Number of sessions included in index file: {len(files_to_use)}')

    # Create index file in dict form
    output_dict = build_index_dict(files_to_use)

    # write out index yaml
    with open(output_file, 'w') as f:
        yaml.safe_dump(output_dict, f)

    return output_file

def aggregate_extract_results_wrapper(input_dir, format, output_dir, mouse_threshold=0.0):
    '''
    Copies all the h5, yaml and avi files generated from all successful extractions to
    a new directory to hold all the necessary data to continue down the MoSeq pipeline.
    Then generates an index file in the base directory/input_dir.

    Parameters
    ----------
    input_dir (str): path to base directory containing all session folders
    format (str): string format for metadata to use as the new aggregated filename
    output_dir (str): name of the directory to create and store all results in
    subpath (str): subdirectory that all sessions must exist within
    mouse_threshold (float): threshold value of mean frame depth to include session frames

    Returns
    -------
    indexpath (str): path to generated index file including all aggregated session information.
    '''

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
    to_load = [tup for tup in to_load if 'sample' not in tup[1]]
    
    loaded = load_h5s(to_load)

    manifest = build_manifest(loaded, format=format)

    copy_manifest_results(manifest, output_dir)

    print('Results successfully aggregated in', output_dir)

    indexpath = generate_index_wrapper(output_dir, os.path.join(input_dir, 'moseq2-index.yaml'), subpath=os.path.dirname(output_dir))

    print(f'Index file path: {indexpath}')
    return indexpath


def get_roi_wrapper(input_file, config_data, output_dir=None):
    '''
    Wrapper function to compute ROI given depth file.

    Parameters
    ----------
    input_file (str): path to depth file.
    config_data (dict): dictionary of ROI extraction parameters.
    output_dir (str): path to desired directory to save results in.

    Returns
    -------
    roi (2d array): ROI image to plot in GUI
    bground_im (2d array): Background image to plot in GUI
    first_frame (2d array): First frame image to plot in GUI
    '''

    if output_dir == None:
        output_dir = os.path.join(os.path.dirname(input_file), 'proc')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # checks camera type to set appropriate bg_roi_weights
    config_data = set_bg_roi_weights(config_data)

    print('Getting background...')
    bground_im = get_bground_im_file(input_file)
    write_image(os.path.join(output_dir, 'bground.tiff'), bground_im, scale=True)

    first_frame = load_movie_data(input_file, 0) # there is a tar object flag that must be set!!
    write_image(os.path.join(output_dir, 'first_frame.tiff'), first_frame, scale=True,
                scale_factor=config_data['bg_roi_depth_range'])

    print('Getting roi...')
    strel_dilate = select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_dilate']))
    strel_erode = select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_erode']))

    rois, plane, _, _, _, _ = get_roi(bground_im,
                                      **config_data,
                                      strel_dilate=strel_dilate,
                                      strel_erode=strel_erode
                                      )

    if config_data['use_plane_bground']:
        print('Using plane fit for background...')
        bground_im = set_bground_to_plane_fit(bground_im, plane, output_dir)

    # Sort ROIs by largest mean area to later select largest one (bg_roi_index)
    if config_data['bg_sort_roi_by_position']:
        rois = rois[:config_data['bg_sort_roi_by_position_max_rois']]
        rois = [rois[i] for i in np.argsort([np.nonzero(roi)[0].mean() for roi in rois])]

    if type(config_data['bg_roi_index']) == int:
        config_data['bg_roi_index'] = [config_data['bg_roi_index']]

    bg_roi_index = [idx for idx in config_data['bg_roi_index'] if idx in range(len(rois))]
    roi = rois[bg_roi_index[0]]

    for idx in bg_roi_index:
        roi_filename = f'roi_{idx:02d}.tiff'
        write_image(os.path.join(output_dir, roi_filename), rois[idx], scale=True, dtype='uint8')

    return roi, bground_im, first_frame

def extract_wrapper(input_file, output_dir, config_data, num_frames=None, skip=False):
    '''
    Wrapper function to run extract function for both GUI and CLI.

    Parameters
    ----------
    input_file (str): path to depth file
    output_dir (str): path to directory to save results in.
    config_data (dict): dictionary containing extraction parameters.
    num_frames (int): number of frames to extract. All if None.
    skip (bool): indicates whether to skip file if already extracted
    extract (function): extraction function state (Only passed by CLI)

    Returns
    -------
    output_dir (str): path to directory containing extraction (only if gui==True)
    '''

    print('Processing:', input_file)
    # get the basic metadata

    status_dict = {
        'parameters': deepcopy(config_data),
        'complete': False,
        'skip': False,
        'uuid': str(uuid.uuid4()),
        'metadata': ''
    }

    # handle tarball stuff
    dirname = os.path.dirname(input_file)

    video_metadata = get_movie_info(input_file)

    # Getting number of frames to extract
    if num_frames is None:
        nframes = int(video_metadata['nframes'])
    elif num_frames > video_metadata['nframes']:
        warnings.warn('Requested more frames than video includes, extracting whole recording...')
        nframes = int(video_metadata['nframes'])
    elif isinstance(num_frames, int):
        nframes = num_frames

    config_data = check_filter_sizes(config_data)

    # If input file is compressed (tarFile), returns decompressed file path and tar bool indicator.
    # Also gets loads respective metadata dictionary and timestamp array.
    input_file, acquisition_metadata, timestamps, alternate_correct, config_data['tar'] = handle_extract_metadata(input_file, dirname)

    status_dict['metadata'] = acquisition_metadata # update status dict

    # Compute total number of frames to include from an initial starting point.
    nframes, first_frame_idx, last_frame_idx = get_frame_range_indices(config_data, nframes)

    # Get specified timestamp range
    if timestamps is not None:
        timestamps = timestamps[first_frame_idx:last_frame_idx]

    # Handle alternative timestamp path time-scale formatting
    if alternate_correct:
        timestamps *= 1000.0

    scalars_attrs = scalar_attributes()
    scalars = list(scalars_attrs.keys())

    # Get frame chunks to extract
    frame_batches = list(gen_batch_sequence(nframes, config_data['chunk_size'], config_data['chunk_overlap'], offset=first_frame_idx))

    # set up the output directory
    if output_dir is None:
        output_dir = os.path.join(dirname, 'proc')
    else:
        if os.path.dirname(output_dir) != dirname:
            output_dir = os.path.join(dirname, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Non-functional indicator for extraction completion statuspytx
    done_file = os.path.join(output_dir, 'done.txt')

    # Ensure index is int
    if isinstance(config_data["bg_roi_index"], list):
        config_data["bg_roi_index"] = config_data["bg_roi_index"][0]

    output_filename = f'results_{config_data["bg_roi_index"]:02d}'
    status_filename = os.path.join(output_dir, f'{output_filename}.yaml')

    # Check if session has already been extracted
    if check_completion_status(status_filename) and skip:
        print('Skipping...')
        return

    # Removing done.txt file if it already exists
    if os.path.exists(done_file):
        os.remove(done_file)

    with open(status_filename, 'w') as f:
        yaml.safe_dump(status_dict, f)

    # Get Structuring Elements for extraction
    str_els = get_strels(config_data)

    # Compute ROIs
    roi, bground_im, first_frame = get_roi_wrapper(input_file, config_data, output_dir=output_dir)

    # Debugging option: DTD has no effect on extraction results unless dilate iterations > 1
    if config_data.get('detected_true_depth', 'auto') == 'auto':
        config_data['true_depth'] = np.median(bground_im[roi > 0])
    else:
        config_data['true_depth'] = int(config_data['detected_true_depth'])

    print('Detected true depth:', config_data['true_depth'])

    if config_data.get('dilate_iterations', 0) > 1:
        print('Dilating background')
        bground_im = graduate_dilated_wall_area(bground_im, config_data, str_els['strel_dilate'], output_dir)

    extraction_data = {
        'bground_im': bground_im,
        'roi': roi,
        'first_frame': first_frame,
        'first_frame_idx': first_frame_idx,
        'nframes': nframes,
        'frame_batches': frame_batches
    }

    # farm out the batches and write to an hdf5 file
    with h5py.File(os.path.join(output_dir, f'{output_filename}.h5'), 'w') as f:
        # Write scalars, roi, acquisition metadata, etc. to h5 file
        create_extract_h5(**extraction_data,
                          h5_file=f,
                          acquisition_metadata=acquisition_metadata,
                          config_data=config_data,
                          status_dict=status_dict,
                          scalars_attrs=scalars_attrs,
                          timestamps=timestamps)

        # Write crop-rotated results to h5 file and write video preview mp4 file
        process_extract_batches(**extraction_data,
                                h5_file=f,
                                input_file=input_file,
                                config_data=config_data,
                                scalars=scalars,
                                str_els=str_els,
                                output_dir=output_dir,
                                output_filename=output_filename)

    print('\n')

    # Compress the depth file to avi format; compresses original raw file by ~8x.
    try:
        if input_file.endswith('dat') and config_data['compress']:
            convert_raw_to_avi_function(input_file,
                                        chunk_size=config_data['compress_chunk_size'],
                                        fps=config_data['fps'],
                                        delete=False, # to be changed when we're ready!
                                        threads=config_data['compress_threads'])
    except AttributeError as e:
        print('Error converting raw video to avi format, continuing anyway...')
        print(e)
        pass

    status_dict['complete'] = True

    with open(status_filename, 'w') as f:
        yaml.safe_dump(status_dict, f)

    with open(done_file, 'w') as f:
        f.write('done')

    return output_dir

def flip_file_wrapper(config_file, output_dir, selected_flip=None):
    '''
    Wrapper function to download and save flip classifiers.

    Parameters
    ----------
    config_file (str): path to config file
    output_dir (str): path to directory to save classifier in.
    selected_flip (int): index of desired flip classifier.

    Returns
    -------
    None
    '''
    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    flip_files = {
        'large mice with fibers':
            "https://storage.googleapis.com/flip-classifiers/flip_classifier_k2_largemicewithfiber.pkl",
        'adult male c57s':
            "https://storage.googleapis.com/flip-classifiers/flip_classifier_k2_c57_10to13weeks.pkl",
        'mice with Inscopix cables':
            "https://storage.googleapis.com/flip-classifiers/flip_classifier_k2_inscopix.pkl"
    }

    key_list = list(flip_files.keys())

    for idx, (k, v) in enumerate(flip_files.items()):
        print(f'[{idx}] {k} ---> {v}')

    # prompt for user selection if not already inputted
    while selected_flip is None:
        try:
            selected_flip = int(input('Enter a selection '))
            if selected_flip > len(flip_files.keys()):
                selected_flip = None
        except ValueError:
            print('Please enter a number')
            continue

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    selection = flip_files[key_list[selected_flip]]

    output_filename = os.path.join(output_dir, os.path.basename(selection))

    urllib.request.urlretrieve(selection, output_filename)
    print('Successfully downloaded flip file to', output_filename)

    # Update the config file with the latest path to the flip classifier
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        config_data['flip_classifier'] = output_filename

        with open(config_file, 'w') as f:
            yaml.safe_dump(config_data, f)
    except Exception as e:
        print('Unexpected error:', e)
        return 'Could not update configuration file flip classifier path'