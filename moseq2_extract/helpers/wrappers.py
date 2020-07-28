'''
Wrapper functions for all functionality afforded by MoSeq2-Extract.
These functions perform all the data processing from start to finish, and are shared between the CLI and GUI.
'''

import os
import sys
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
                            copy_manifest_results, build_index_dict, check_completion_status, get_pca_uuids
from moseq2_extract.util import select_strel, gen_batch_sequence, scalar_attributes, convert_raw_to_avi_function, \
                        set_bground_to_plane_fit, recursive_find_h5s, clean_dict, graduate_dilated_wall_area, \
                        h5_to_dict, set_bg_roi_weights, get_frame_range_indices


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


def generate_index_wrapper(input_dir, pca_file, output_file, filter, all_uuids, subpath='/proc/'):
    '''
    Generates index file containing a summary of all extracted sessions.

    Parameters
    ----------
    input_dir (str): directory to search for extracted sessions.
    pca_file (str): path to pca_scores file.
    output_file (str): preferred name of the index file.
    filter (list): list of metadata keys to conditionally filter.
    all_uuids (list): list of all extracted session uuids.
    subpath (str): subdirectory that aggregated files must contain.

    Returns
    -------
    output_file (str): path to index file.
    '''

    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # gather than h5s and the pca scores file
    # uuids should match keys in the scores file
    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    # Checking for a pre-existing PCA scores file to load
    # only the files that the PCA was trained on.
    # If pca doesn't exist, then all_uuids is returned.
    pca_uuids = get_pca_uuids(dicts, pca_file, all_uuids)

    try:
        file_with_uuids = [(os.path.abspath(h5), os.path.abspath(yml), meta) for h5, yml, meta in
                           zip(h5s, yamls, dicts) if meta['uuid'] in pca_uuids]
    except:
        file_with_uuids = [(os.path.abspath(h5), os.path.abspath(yml), meta) for h5, yml, meta in
                           zip(h5s, yamls, dicts)]

    # Ensuring all retrieved extracted session h5s have the appropriate metadata
    # included in their results_00.h5 file
    try:
        if 'metadata' not in file_with_uuids[0][2]:
            for h5 in h5s:
                copy_h5_metadata_to_yaml_wrapper(input_dir, h5)
            file_with_uuids = [(os.path.abspath(h5), os.path.abspath(yml), meta) for h5, yml, meta in
                               zip(h5s, yamls, dicts) if meta['uuid'] in pca_uuids]
    except:
        print('Metadata not found, creating minimal Index file.')

    # Filtering out sessions that do not contain the required subpath in their paths.
    # Ensures that there are no sample extractions included in the index file.
    keep_samples = [i for i, f in enumerate(file_with_uuids) if subpath in f[0]]
    files_to_use = [tup for i, tup in enumerate(file_with_uuids) if i in keep_samples]

    print(f'Number of sessions included in index file: {len(files_to_use)}')

    # Create index file in dict form
    output_dict = build_index_dict(filter, files_to_use, pca_file)

    # write out index yaml
    with open(output_file, 'w') as f:
        yaml.safe_dump(output_dict, f)

    return output_file

def aggregate_extract_results_wrapper(input_dir, format, output_dir):
    '''
    Copies all the h5, yaml and avi files generated from all successful extractions to
    a new directory to hold all the necessary data to continue down the moseq pipeline.

    Parameters
    ----------
    input_dir (str): path to base directory containing all session folders
    format (str): string format for metadata to use as the new aggregated filename
    output_dir (str): name of the directory to create and store all results in

    Returns
    -------
    None
    '''

    mouse_threshold = 0 # defaulting this for now
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


def get_roi_wrapper(input_file, config_data, output_dir=None, gui=False, extract_helper=False):
    '''
    Wrapper function to compute ROI given depth file.

    Parameters
    ----------
    input_file (str): path to depth file.
    config_data (dict): dictionary of ROI extraction parameters.
    output_dir (str): path to desired directory to save results in.
    gui (bool): indicate whether GUI is running.
    extract_helper (bool): indicate whether this is being run independently or by extract function

    Returns
    -------
    if gui:
        output_dir (str): path to saved ROI results
    elif extract_helper:
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
    strel_erode = select_strel(config_data['bg_roi_shape'], tuple(config_data['strel_erode']))

    rois, plane, _, _, _, _ = get_roi(bground_im,
                                  strel_dilate=strel_dilate,
                                  dilate_iters=config_data['dilate_iterations'],
                                  erode_iters=config_data['erode_iterations'],
                                  strel_erode=strel_erode if config_data['erode_iterations'] > 0 else None,
                                  noise_tolerance=config_data['noise_tolerance'],
                                  weights=config_data['bg_roi_weights'],
                                  depth_range=config_data['bg_roi_depth_range'],
                                  overlap_roi=config_data.get('overlap_roi'),
                                  gradient_filter=config_data['bg_roi_gradient_filter'],
                                  gradient_threshold=config_data['bg_roi_gradient_threshold'],
                                  gradient_kernel=config_data['bg_roi_gradient_kernel'],
                                  fill_holes=config_data['bg_roi_fill_holes'])

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

    if gui:
        return output_dir # GUI
    if extract_helper:
        return roi, bground_im, first_frame # HELPER

def extract_wrapper(input_file, output_dir, config_data, num_frames=None, skip=False, gui=False):
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
    gui (bool): indicates if GUI is running.

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
    if num_frames == None:
        nframes = int(video_metadata['nframes'])
    else:
        nframes = num_frames
        if nframes > int(video_metadata['nframes']):
            warnings.warn('Requested more frames than video includes, extracting whole recording...')
            nframes = int(video_metadata['nframes'])

    # If input file is compressed (tarFile), returns decompressed file path and tar bool indicator.
    # Also gets loads respective metadata dictionary and timestamp array.
    input_file, acquisition_metadata, timestamps, alternate_correct, tar = handle_extract_metadata(input_file, dirname)

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
    frame_batches = list(gen_batch_sequence(nframes, config_data['chunk_size'], config_data['chunk_overlap']))

    # set up the output directory
    if output_dir is None:
        output_dir = os.path.join(dirname, 'proc')
    else:
        if os.path.dirname(output_dir) != dirname:
            output_dir = os.path.join(dirname, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = f'results_{config_data["bg_roi_index"]:02d}'
    status_filename = os.path.join(output_dir, f'{output_filename}.yaml')

    if check_completion_status(status_filename):
        if gui and skip: # Skipping already extracted session
            print('Skipping...')
            return
        elif not gui:
            overwrite = input('Press ENTER to overwrite your previous extraction, else to end the process.')
            if overwrite != '':
                sys.exit(0)

    with open(status_filename, 'w') as f:
        yaml.safe_dump(status_dict, f)

    strel_dilate = select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_dilate']))
    strel_tail = select_strel((config_data['tail_filter_shape'], config_data['tail_filter_size']))
    strel_min = select_strel((config_data['cable_filter_shape'], config_data['cable_filter_size']))

    roi, bground_im, first_frame = get_roi_wrapper(input_file, config_data, output_dir=output_dir, extract_helper=True)

    # Debugging option: DTD has no effect on extraction results unless dilate iterations > 1
    if config_data.get('detected_true_depth', 'auto') == 'auto':
        true_depth = np.median(bground_im[roi > 0])
    else:
        true_depth = int(config_data['detected_true_depth'])

    print('Detected true depth:', true_depth)

    if config_data.get('dilate_iterations', 0) > 1:
        print('Dilating background')
        bground_im = graduate_dilated_wall_area(bground_im, config_data, strel_dilate, true_depth, output_dir)

    # farm out the batches and write to an hdf5 file
    with h5py.File(os.path.join(output_dir, f'{output_filename}.h5'), 'w') as f:
        # Write scalars, roi, acquisition metadata, etc. to h5 file
        create_extract_h5(f, acquisition_metadata, config_data, status_dict, scalars, scalars_attrs, nframes,
                          true_depth, roi, bground_im, first_frame, timestamps)

        # Generate crop-rotated result
        video_pipe = process_extract_batches(f, input_file, config_data, bground_im, roi, scalars, frame_batches,
                                 first_frame_idx, true_depth, tar, strel_tail, strel_min, output_dir, output_filename)

        if video_pipe:
            video_pipe.stdin.close()
            video_pipe.wait()

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

    with open(os.path.join(output_dir, 'done.txt'), 'w') as f:
        f.write('done')

    if gui:
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
        selected_flip = int(input('Enter a selection '))
        if selected_flip > len(flip_files.keys()):
            selected_flip = None

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

    except:
        print('Unexpected error:', sys.exc_info()[0])
        return 'Could not update configuration file flip classifier path'