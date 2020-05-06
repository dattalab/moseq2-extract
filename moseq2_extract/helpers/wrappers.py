import os
import re
import sys
import uuid
import h5py
from tqdm.auto import tqdm
import urllib.request
import shutil
import warnings
import numpy as np
from cytoolz import pluck
from copy import deepcopy
import ruamel.yaml as yaml
from moseq2_extract.io.image import write_image
from moseq2_extract.helpers.extract import process_extract_batches
from moseq2_extract.extract.proc import get_roi, get_bground_im_file
from moseq2_extract.helpers.data import handle_extract_metadata, create_extract_h5
from moseq2_extract.io.video import load_movie_data, convert_mkv_to_avi, get_movie_info
from moseq2_extract.util import select_strel, gen_batch_sequence, load_metadata, \
                            load_timestamps, convert_raw_to_avi_function, scalar_attributes, recursive_find_h5s, \
                            clean_dict, h5_to_dict, graduate_dilated_wall_area


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


def generate_index_wrapper(input_dir, pca_file, output_file, filter, all_uuids):
    '''
    Generates index file containing a summary of all extracted sessions.

    Parameters
    ----------
    input_dir (str): directory to search for extracted sessions.
    pca_file (str): path to pca_scores file.
    output_file (str): preferred name of the index file.
    filter (list): list of metadata keys to conditionally filter.
    all_uuids (list): list of all session uuids.

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
    if not os.path.exists(pca_file) or all_uuids:
        warnings.warn('Will include all files')
        pca_uuids = [dct['uuid'] for dct in dicts]
    else:
        if not os.path.exists(pca_file) or all_uuids:
            warnings.warn('Will include all files')
            pca_uuids = pluck('uuid', dicts)
        else:
            with h5py.File(pca_file, 'r') as f:
                pca_uuids = list(f['scores'])
    try:
        file_with_uuids = [(os.path.abspath(h5), os.path.abspath(yml), meta) for h5, yml, meta in
                           zip(h5s, yamls, dicts) if meta['uuid'] in pca_uuids]
    except:
        file_with_uuids = [(os.path.abspath(h5), os.path.abspath(yml), meta) for h5, yml, meta in
                           zip(h5s, yamls, dicts)]
    try:
        if 'metadata' not in file_with_uuids[0][2]:
            #raise RuntimeError('Metadata not present in yaml files, run copy-h5-metadata-to-yaml to update yaml files')
            for h5 in h5s:
                copy_h5_metadata_to_yaml_wrapper(input_dir, h5)
            file_with_uuids = [(os.path.abspath(h5), os.path.abspath(yml), meta) for h5, yml, meta in
                               zip(h5s, yamls, dicts) if meta['uuid'] in pca_uuids]
    except:
        print('Metadata not found, creating minimal Index file.')

    output_dict = {
        'files': [],
        'pca_path': pca_file
    }

    index_uuids = []
    for i, file_tup in enumerate(file_with_uuids):
        if file_tup[2]['uuid'] not in index_uuids:
            try:
                output_dict['files'].append({
                    'path': (file_tup[0], file_tup[1]),
                    'uuid': file_tup[2]['uuid'],
                    'group': 'default'
                })
                index_uuids.append(file_tup[2]['uuid'])

                output_dict['files'][i]['metadata'] = {}

                for k, v in file_tup[2]['metadata'].items():
                    for filt in filter:
                        if k == filt[0]:
                            tmp = re.match(filt[1], v)
                            if tmp is not None:
                                v = tmp[0]

                    output_dict['files'][i]['metadata'][k] = v
            except:
                pass

    # write out index yaml
    with open(output_file, 'w') as f:
        yaml.safe_dump(output_dict, f)

    return output_file


def get_roi_wrapper(input_file, config_data, output_dir=None, output_directory=None, gui=False, extract_helper=False):
    '''
    Wrapper function to compute ROI given depth file.

    Parameters
    ----------
    input_file (str): path to depth file.
    config_data (dict): dictionary of ROI extraction parameters.
    output_dir (str): path to desired directory to save results in.
    output_directory (str): GUI optional secondary external save directory path
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
    if gui:
        if output_directory is not None:
            output_dir = os.path.join(output_directory, 'proc')
        else:
            output_dir = os.path.join(os.path.dirname(input_file), 'proc')
    else:
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(input_file), 'proc')

    if type(config_data['bg_roi_index']) is int:
        bg_roi_index = [config_data['bg_roi_index']]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Getting background...')
    bground_im = get_bground_im_file(input_file)
    write_image(os.path.join(output_dir, 'bground.tiff'), bground_im, scale=True)

    first_frame = load_movie_data(input_file, 0) # there is a tar object flag that must be set!!
    write_image(os.path.join(output_dir, 'first_frame.tiff'), first_frame, scale=True,
                scale_factor=config_data['bg_roi_depth_range'])

    print('Getting roi...')
    strel_dilate = select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_dilate']))

    rois, plane, _, _, _, _ = get_roi(bground_im,
                                  strel_dilate=strel_dilate,
                                  dilate_iters=config_data['dilate_iterations'],
                                  weights=config_data['bg_roi_weights'],
                                  depth_range=config_data['bg_roi_depth_range'],
                                  gradient_filter=config_data['bg_roi_gradient_filter'],
                                  gradient_threshold=config_data['bg_roi_gradient_threshold'],
                                  gradient_kernel=config_data['bg_roi_gradient_kernel'],
                                  fill_holes=config_data['bg_roi_fill_holes'], gui=gui)

    if config_data['use_plane_bground']:
        print('Using plane fit for background...')
        xx, yy = np.meshgrid(np.arange(bground_im.shape[1]), np.arange(bground_im.shape[0]))
        coords = np.vstack((xx.ravel(), yy.ravel()))
        plane_im = (np.dot(coords.T, plane[:2]) + plane[3]) / -plane[2]
        plane_im = plane_im.reshape(bground_im.shape)
        write_image(os.path.join(output_dir, 'bground.tiff'), plane_im, scale=True)
        bground_im = plane_im

    if config_data['bg_sort_roi_by_position']:
        rois = rois[:config_data['bg_sort_roi_by_position_max_rois']]
        rois = [rois[i] for i in np.argsort([np.nonzero(roi)[0].mean() for roi in rois])]

    if type(config_data['bg_roi_index']) == int:
        config_data['bg_roi_index'] = [config_data['bg_roi_index']]

    bg_roi_index = [idx for idx in config_data['bg_roi_index'] if idx in range(len(rois))]
    roi = rois[bg_roi_index[0]]

    for idx in bg_roi_index:
        roi_filename = f'roi_{idx:02d}.tiff'
        write_image(os.path.join(output_dir, roi_filename),
                    rois[idx], scale=True, dtype='uint8')

    if gui:
        return output_dir # GUI
    if extract_helper:
        return roi, bground_im, first_frame # HELPER

def extract_wrapper(input_file, output_dir, config_data, num_frames=None, skip=False, extract=None, gui=False):
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


    if config_data['spatial_filter_size'][0] % 2 == 0 and config_data['spatial_filter_size'][0] > 0:
        config_data['spatial_filter_size'][0] += 1
    if config_data['temporal_filter_size'][0] % 2 == 0 and config_data['temporal_filter_size'][0] > 0:
        config_data['temporal_filter_size'][0] += 1

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
    if num_frames == None:
        nframes = int(video_metadata['nframes'])
    else:
        if num_frames > int(video_metadata['nframes']):
            print('Requested more frames than video includes, extracting whole recording...')
            nframes = int(video_metadata['nframes'])
        else:
            nframes = num_frames

    metadata_path, timestamp_path, alternate_correct, tar, \
    nframes, first_frame_idx, last_frame_idx = handle_extract_metadata(input_file, dirname, config_data, nframes)

    acquisition_metadata = load_metadata(metadata_path)
    status_dict['metadata'] = acquisition_metadata
    timestamps = load_timestamps(timestamp_path, col=0)

    if timestamps is not None:
        timestamps = timestamps[first_frame_idx:last_frame_idx]

    if alternate_correct:
        timestamps *= 1000.0

    scalars_attrs = scalar_attributes()
    scalars = list(scalars_attrs.keys())

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

    if not gui:
        # CLI FUNCTIONALITY
        if os.path.exists(status_filename):
            overwrite = input('Press ENTER to overwrite your previous extraction, else to end the process.')
            if overwrite != '':
                sys.exit(0)
    else:
        # GUI FUNCTIONALITY
        if skip == True:
            if os.path.exists(os.path.join(output_dir, 'done.txt')):
                return

    with open(status_filename, 'w') as f:
        yaml.safe_dump(status_dict, f)

    bg_roi_file = input_file

    strel_dilate = select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_dilate']))
    strel_tail = select_strel((config_data['tail_filter_shape'], config_data['tail_filter_size']))
    strel_min = select_strel((config_data['cable_filter_shape'], config_data['cable_filter_size']))

    roi, bground_im, first_frame = get_roi_wrapper(bg_roi_file, config_data, output_dir=output_dir, extract_helper=True)

    if config_data.get('detected_true_depth', 'auto') == 'auto':
        true_depth = np.median(bground_im[roi > 0])
    else:
        true_depth = int(config_data['detected_true_depth'])

    if config_data.get('dilate_iterations', 0) > 1:
        print('Dilating background')
        bground_im = graduate_dilated_wall_area(bground_im, config_data, strel_dilate, true_depth, output_dir)

    print('Detected true depth:', true_depth)

    # farm out the batches and write to an hdf5 file
    with h5py.File(os.path.join(output_dir, f'{output_filename}.h5'), 'w') as f:

        create_extract_h5(f, acquisition_metadata, config_data, status_dict, scalars, scalars_attrs, nframes,
                          true_depth, roi, bground_im, first_frame, timestamps, extract=extract)

        video_pipe = process_extract_batches(f, input_file, config_data, bground_im, roi, scalars, frame_batches,
                                             first_frame_idx, true_depth, tar, strel_tail, strel_min, output_dir, output_filename)

        if video_pipe:
            video_pipe.stdin.close()
            video_pipe.wait()

    print('\n')

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

def flip_file_wrapper(config_file, output_dir, selected_flip=1, gui=False):
    '''
    Wrapper function to download and save flip classifiers.
    Parameters
    ----------
    config_file (str): path to config file
    output_dir (str): path to directory to save classifier in.
    selected_flip (int): index of desired flip classifier.
    gui (bool): indicates if the GUI is running.
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

    selection = None
    key_list = list(flip_files.keys())

    if gui:
        selection = selected_flip
    else:
        for idx, (k, v) in enumerate(flip_files.items()):
            print(f'[{idx}] {k} ---> {v}')

        while selection is None:
            selection = int(input('Enter a selection '))
            if selection > len(flip_files.keys()):
                selection = None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    selection = flip_files[key_list[selection]]

    output_filename = os.path.join(output_dir, os.path.basename(selection))
    if gui:
        if os.path.exists(output_filename):
            print('This file already exists, would you like to overwrite it? [Y -> yes, else -> exit]')
            ow = input()
            if ow == 'Y':
                urllib.request.urlretrieve(selection, output_filename)
                print('Successfully downloaded flip file to', output_filename)
            else:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                f.close()
                return f'Retained older flip file version: {config_data["flip_classifier"]}'
        else:
            urllib.request.urlretrieve(selection, output_filename)
            print('Successfully downloaded flip file to', output_filename)
    else:
        urllib.request.urlretrieve(selection, output_filename)
        print('Successfully downloaded flip file to', output_filename)

    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        f.close()
        config_data['flip_classifier'] = output_filename
        with open(config_file, 'w') as f:
            yaml.safe_dump(config_data, f)
        f.close()
    except:
        print('Unexpected error:', sys.exc_info()[0])
        return 'Could not update configuration file flip classifier path'