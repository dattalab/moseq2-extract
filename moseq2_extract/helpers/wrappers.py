import os
import sys
import uuid
import h5py
import urllib
import warnings
import numpy as np
from copy import deepcopy
import ruamel.yaml as yaml
from moseq2_extract.io.image import write_image
from moseq2_extract.helpers.extract import process_extract_batches
from moseq2_extract.extract.proc import get_roi, get_bground_im_file
from moseq2_extract.helpers.data import handle_extract_metadata, create_extract_h5
from moseq2_extract.io.video import load_movie_data, convert_mkv_to_avi, get_movie_info
from moseq2_extract.util import select_strel, gen_batch_sequence, load_metadata, \
                            load_timestamps, convert_raw_to_avi_function, scalar_attributes

def get_roi_wrapper(input_file, config_data, output_dir=None, output_directory=None, gui=False, extract_helper=False):

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

    # NEW FUNCTIONALITY S
    if input_file.endswith('.mkv'):
        # create a depth.avi file to represent PCs
        temp = convert_mkv_to_avi(input_file)
        if isinstance(temp, str):
            input_file = temp
    # NEW FUNCTIONALITY E

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
        roi_filename = 'roi_{:02d}.tiff'.format(idx)
        write_image(os.path.join(output_dir, roi_filename),
                    rois[idx], scale=True, dtype='uint8')

    if gui:
        return output_dir # GUI
    if extract_helper:
        return roi, bground_im, first_frame # HELPER

def extract_wrapper(input_file, output_dir, config_data, num_frames=None, skip=False, extract=None, gui=False):
    # get the background and roi, which will be used across all batches

    ## NEW FUNCTIONALITY S
    if config_data['spatial_filter_size'][0] % 2 == 0 and config_data['spatial_filter_size'][0] > 0:
        config_data['spatial_filter_size'][0] += 1
    if config_data['temporal_filter_size'][0] % 2 == 0 and config_data['temporal_filter_size'][0] > 0:
        config_data['temporal_filter_size'][0] += 1
    # NEW FUNCTIONALITY E

    print('Processing: {}'.format(input_file))
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
        output_dir = os.path.join(dirname, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = 'results_{:02d}'.format(config_data['bg_roi_index'])
    status_filename = os.path.join(output_dir, '{}.yaml'.format(output_filename))

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
    if input_file.endswith('.mkv'):
        # create a depth.avi file to represent PCs
        bg_roi_file = convert_mkv_to_avi(input_file)

    strel_tail = select_strel((config_data['tail_filter_shape'], config_data['tail_filter_size']))
    strel_min = select_strel((config_data['cable_filter_shape'], config_data['cable_filter_size']))

    if bg_roi_file != input_file:
        roi, bground_im, first_frame = get_roi_wrapper(bg_roi_file, config_data,
                                                      output_dir=output_dir, extract_helper=True)
    else:
        roi, bground_im, first_frame = get_roi_wrapper(input_file, config_data,
                                                      output_dir=output_dir, extract_helper=True)

    if config_data['detected_true_depth'] == 'auto':
        true_depth = np.median(bground_im[roi > 0])
    else:
        true_depth = int(config_data['detected_true_depth'])

    if input_file.endswith('mkv'):
        new_bg = np.ma.masked_not_equal(roi, 0)
        bground_im = np.where(new_bg == True, new_bg, true_depth)

    print('Detected true depth: {}'.format(true_depth))

    # farm out the batches and write to an hdf5 file
    with h5py.File(os.path.join(output_dir, '{}.h5'.format(output_filename)), 'w') as f:

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
        pass

    status_dict['complete'] = True

    with open(status_filename, 'w') as f:
        yaml.safe_dump(status_dict, f)

    with open(os.path.join(output_dir, 'done.txt'), 'w') as f:
        f.write('done')

    print(f'Sample extraction of {str(nframes)} frames completed successfully in {output_dir}.')
    if gui:
        return output_dir

def flip_file_wrapper(config_file, output_dir, selected_flip=1, gui=False):
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
        for i, f in enumerate(flip_files):
            print(f'[{i}] {f}')

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
                print('Successfully downloaded flip file to {}'.format(output_filename))
            else:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                f.close()
                return 'Retained older flip file version: {}'.format(config_data['flip_classifier'])
        else:
            urllib.request.urlretrieve(selection, output_filename)
            print('Successfully downloaded flip file to {}'.format(output_filename))
    else:
        urllib.request.urlretrieve(selection, output_filename)
        print('Successfully downloaded flip file to {}'.format(output_filename))

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