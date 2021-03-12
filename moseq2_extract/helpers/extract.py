'''

Extraction-helper utilities.
These functions are primarily called from inside the extract_wrapper() function.

'''

import numpy as np
from tqdm.auto import tqdm
from moseq2_extract.extract.extract import extract_chunk
from moseq2_extract.io.video import load_movie_data, write_frames_preview

def write_extracted_chunk_to_h5(h5_file, results, config_data, scalars, frame_range, offset, depth_mapping=True):
    '''

    Write extracted frames, frame masks, and scalars to an open h5 file.

    Parameters
    ----------
    h5_file (H5py.File): open results_00 h5 file to save data in.
    results (dict): extraction results dict.
    config_data (dict): dictionary containing extraction parameters (autogenerated)
    scalars (list): list of keys to scalar attribute values
    frame_range (range object): current chunk frame range
    offset (int): frame offset

    Returns
    -------
    '''

    if depth_mapping:
        # Writing computed scalars to h5 file
        for scalar in scalars:
            h5_file[f'scalars/{scalar}'][frame_range] = results['scalars'][scalar][offset:]

        # Writing frames and mask to h5
        h5_file['frames'][frame_range] = results['depth_frames'][offset:]
        h5_file['frames_mask'][frame_range] = results['mask_frames'][offset:]

        # Writing flip classifier results to h5
        if config_data['flip_classifier']:
            h5_file['metadata/extraction/flips'][frame_range] = results['flips'][offset:]
    else:
        # Writing alternative mapping frames to h5
        h5_file['alt_frames'][frame_range] = results['alt_frames'][offset:]

def set_tracking_model_parameters(results, config_data):
    '''
    Helper function to threshold and clip the masked frame data if use_tracking_model = True.
    Updates the tracking_init_mean and tracking_init_cov variables in config_data.

    Parameters
    ----------
    results (dict): dict of extracted depth frames and mask frames to threshold to update.
    config_data (dict): dict of config parameters being used to extract current input file.

    Returns
    -------
    results (dict): updated results dict with thresholded and clipped mask_frames.
    config_data (dict): updated config data parameter dict
    '''

    # Thresholding and clipping EM-tracked frame mask data
    results['mask_frames'][results['depth_frames'] < config_data['min_height']] = config_data[
        'tracking_model_ll_clip']
    results['mask_frames'][results['mask_frames'] < config_data['tracking_model_ll_clip']] = config_data[
        'tracking_model_ll_clip']
    # Updating EM tracking estimators
    config_data['tracking_init_mean'] = results['parameters']['mean'][-(config_data['chunk_overlap'] + 1)]
    config_data['tracking_init_cov'] = results['parameters']['cov'][-(config_data['chunk_overlap'] + 1)]

    return results, config_data

def make_output_movie(results, config_data, offset=0):
    '''
    Creates an array for output movie with filtered video and cropped mouse on the top left

    Parameters
    ----------
    results (dict): dict of extracted depth frames, and original raw chunk to create an output movie.
    config_data (dict): dict of extraction parameters containing the crop sizes used in the extraction.
    offset (int): current offset being used, automatically set if chunk_overlap > 0

    Returns
    -------
    output_movie (3D np.array): output movie to write to mp4 file; dims = (nframes, rows, cols).
    '''

    # Create empty array for output movie with filtered video and cropped mouse on the top left
    nframes, rows, cols = results['chunk'][offset:].shape
    output_movie = np.zeros((nframes, rows + config_data['crop_size'][0], cols + config_data['crop_size'][1]),
                            'uint16')

    # Populating array with filtered and cropped videos
    output_movie[:, :config_data['crop_size'][0], :config_data['crop_size'][1]] = results['depth_frames'][offset:]
    output_movie[:, config_data['crop_size'][0]:, config_data['crop_size'][1]:] = results['chunk'][offset:]

    return output_movie


def process_extract_batches(input_file, config_data, bground_im, roi,
                            frame_batches, str_els, output_mov_path,
                            scalars=None, h5_file=None, **kwargs):
    '''
    Compute extracted frames and save them to h5 files and avi files.
    Given an open h5 file, which is used to store extraction results, and some pre-computed input session data points
    such as the background, ROI, etc.
    Called from extract_wrapper()

    Parameters
    ----------
    h5file (h5py.File): opened h5 file to write extracted batches to
    input_file (str): path to depth file
    config_data (dict): dictionary containing extraction parameters (autogenerated)
    bground_im (2d numpy array):  r x c, background image
    roi (2d numpy array):  r x c, roi image
    scalars (list): list of keys to scalar attribute values
    frame_batches (list): list of batches of frames to serially process.
    str_els (dict): dictionary containing OpenCV StructuringElements
    output_mov_path (str): path and filename of the output movie generated by the extraction
    kwargs (dict): Extra keyword arguments.

    Returns
    -------
    config_data (dict): dictionary containing updated extraction validation parameter values
    '''

    video_pipe = None
    config_data['tracking_init_mean'] = None
    config_data['tracking_init_cov'] = None

    extractable = (config_data.get('mapping', 'DEPTH') == 'DEPTH') or (config_data.get('mapping', 0) == 0)

    for i, frame_range in enumerate(tqdm(frame_batches, desc='Processing batches')):
        raw_chunk = load_movie_data(input_file,
                                    frame_range,
                                    frame_size=bground_im.shape[::-1],
                                    **config_data)

        offset = config_data['chunk_overlap'] if i > 0 else 0

        if extractable:
            # Get crop-rotated frame batch
            results = extract_chunk(**config_data,
                                    **str_els,
                                    chunk=raw_chunk,
                                    roi=roi,
                                    bground=bground_im
                                    )

            if config_data['use_tracking_model']:
                # threshold and clip mask frames from EM tracking results
                results, config_data = set_tracking_model_parameters(results, config_data)
        else:
            # mapping is a color or IR image
            results = {'alt_frames': raw_chunk}

        # Offsetting frame chunk by CLI parameter defined option: chunk_overlap
        frame_range = frame_range[offset:]

        if h5_file is not None:
            write_extracted_chunk_to_h5(h5_file, results, config_data, scalars, frame_range, offset, extractable)

        if extractable:
            # Create array for output movie with filtered video and cropped mouse on the top left
            output_movie = make_output_movie(results, config_data, offset)

            # Writing frame batch to mp4 file
            video_pipe = write_frames_preview(output_mov_path, output_movie,
                pipe=video_pipe, close_pipe=False, fps=config_data['fps'],
                frame_range=list(frame_range),
                depth_max=config_data['max_height'], depth_min=config_data['min_height'],
                progress_bar=config_data.get('progress_bar', False))

    # Check if video is done writing. If not, wait.
    if video_pipe is not None:
        video_pipe.stdin.close()
        video_pipe.wait()

def run_local_extract(to_extract, config_file, skip_extracted=False):
    '''
    Runs the extract command on given list of sessions to extract on a local platform.
    This function is meant for the GUI interface to utilize the moseq2-batch extract functionality.

    Parameters
    ----------
    to_extract (list): list of paths to files to extract
    config_file (str): path to configuration file containing pre-configured extract and ROI
    skip_extracted (bool): Whether to skip already extracted session.

    Returns
    -------
    None
    '''
    from moseq2_extract.gui import extract_command

    for ext in tqdm(to_extract, desc='Extracting Sessions'):
        try:
            extract_command(ext, None, config_file=config_file, skip=skip_extracted)
        except Exception as e:
            print('Unexpected error:', e)
            print('could not extract', ext)