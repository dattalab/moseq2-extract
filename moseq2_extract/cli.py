'''
CLI front-end operations. This module contains all the functionality and configurable parameters
users can alter to most accurately process their data.

Note: These functions simply read all the parameters into a dictionary,
 and then call the corresponding wrapper function with the given input parameters.
'''

import os
import sys
import click
import numpy as np
import ruamel.yaml as yaml
from tqdm.auto import tqdm
from moseq2_extract.util import (gen_batch_sequence, command_with_config)
from moseq2_extract.io.video import (get_movie_info, load_movie_data, write_frames)
from moseq2_extract.helpers.wrappers import get_roi_wrapper, extract_wrapper, flip_file_wrapper, \
                                            generate_index_wrapper, aggregate_extract_results_wrapper

orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


@click.group()
@click.version_option()
def cli():
    pass

def common_roi_options(function):
    '''
    Decorator function for grouping shared ROI related parameters.
    The parameters included in this function are shared between the find_roi and extract CLI commands.

    Parameters
    ----------
    function: Function to add enclosed parameters to as click options.

    Returns
    -------
    function: Updated function including shared parameters.
    '''

    function = click.option('--bg-roi-dilate', default=(10, 10), type=(int, int),
                            help='Size of strel to dilate roi')(function)
    function = click.option('--bg-roi-shape', default='ellipse', type=str,
                            help='Shape to use to dilate roi (ellipse or rect)')(function)
    function = click.option('--bg-roi-index', default=0, type=int,
                            help='Index of which background mask(s) to use')(function)
    function = click.option('--bg-roi-weights', default=(1, .1, 1), type=(float, float, float),
                            help='ROI feature weighting (area, extent, dist)')(function)
    function = click.option('--camera-type', default='kinect', type=click.Choice(["kinect", "azure", "realsense"]),
                            help='Helper parameter: auto-sets bg-roi-weights to precomputed values for different camera types. \
                             Possible types: ["kinect", "azure", "realsense"]')(function)
    function = click.option('--bg-roi-depth-range', default=(650, 750), type=(float, float),
                            help='Range to search for floor of arena (in mm)')(function)
    function = click.option('--bg-roi-gradient-filter', default=False, type=bool,
                            help='Exclude walls with gradient filtering')(function)
    function = click.option('--bg-roi-gradient-threshold', default=3000, type=float,
                            help='Gradient must be < this to include points')(function)
    function = click.option('--bg-roi-gradient-kernel', default=7, type=int,
                            help='Kernel size for Sobel gradient filtering')(function)
    function = click.option('--bg-roi-fill-holes', default=True, type=bool, help='Fill holes in ROI')(function)
    function = click.option('--bg-sort-roi-by-position', default=False, type=bool,
                            help='Sort ROIs by position')(function)
    function = click.option('--bg-sort-roi-by-position-max-rois', default=2, type=int,
                            help='Max original ROIs to sort by position')(function)
    function = click.option('--dilate-iterations', default=0, type=int,
                            help='Number of dilation iterations to increase bucket floor size. (Special Cases Only)')(function)
    function = click.option('--bg-roi-erode', default=(1, 1), type=(int, int),
                            help='Size of cv2 Structure Element to erode roi. (Special Cases Only)')(function)
    function = click.option('--erode-iterations', default=0, type=int,
                            help='Number of erosion iterations to decrease bucket floor size. (Special Cases Only)')(function)
    function = click.option('--noise-tolerance', default=30, type=int,
                            help='Extent of noise to accept during RANSAC Plane ROI computation. (Special Cases Only)')(function)
    function = click.option('--output-dir', default=None, help='Output directory to save the results h5 file')(function)
    function = click.option('--use-plane-bground', is_flag=True,
                            help='Use a plane fit for the background. Useful for mice that don\'t move much')(function)
    function = click.option("--config-file", type=click.Path())(function)
    function = click.option('--progress-bar', '-p', is_flag=True, help='Show verbose progress bars.')(function)
    return function

def common_avi_options(function):
    '''
    Decorator function for grouping shared video processing parameters.
    The included parameters are shared between convert_raw_to_avi() and copy_slice()

    Parameters
    ----------
    function: Function to add enclosed parameters to as click options.

    Returns
    -------
    function: Updated function including shared parameters.
    '''

    function = click.option('-o', '--output-file', type=click.Path(), default=None, help='Path to output file')(function)
    function = click.option('-b', '--chunk-size', type=int, default=3000, help='Chunk size')(function)
    function = click.option('--fps', type=float, default=30, help='Video FPS')(function)
    function = click.option('--delete', is_flag=True, help='Delete raw file if encoding is sucessful')(function)
    function = click.option('-t', '--threads', type=int, default=3, help='Number of threads for encoding')(function)
    return function



@cli.command(name="find-roi", cls=command_with_config('config_file'), help="Finds the ROI and background distance to subtract from frames when extracting.")
@click.argument('input-file', type=click.Path(exists=True))
@common_roi_options
def find_roi(input_file, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, camera_type, bg_roi_depth_range,
             bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes,
             bg_sort_roi_by_position, bg_sort_roi_by_position_max_rois, dilate_iterations, bg_roi_erode,
             erode_iterations, noise_tolerance, output_dir, use_plane_bground, config_file, progress_bar):

    click_data = click.get_current_context().params
    get_roi_wrapper(input_file, click_data, output_dir)

@cli.command(name="extract", cls=command_with_config('config_file'), help="Processes raw input depth recordings to output a cropped and oriented\
                                            video of the mouse and saves the output+metadata to h5 files in the given output directory.")
@click.argument('input-file', type=click.Path(exists=True, resolve_path=True))
@common_roi_options
@click.option('--crop-size', '-c', default=(80, 80), type=(int, int), help='Width and height of cropped mouse image')
@click.option('--min-height', default=10, type=int, help='Min mouse height from floor (mm)')
@click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
@click.option('--detected-true-depth', default='auto', type=str, help='Option to override automatic depth estimation during extraction. \
            This is only a debugging parameter, for cases where dilate_iterations > 1, otherwise has no effect. Either "auto" or an int value.')
@click.option('--compute-raw-scalars', is_flag=True, help="Compute scalar values from raw cropped frames.")
@click.option('--fps', default=30, type=int, help='Frame rate of camera')
@click.option('--flip-classifier', default=None, help='Location of the flip classifier used to properly orient the mouse (.pkl file)')
@click.option('--flip-classifier-smoothing', default=51, type=int, help='Number of frames to smooth flip classifier probabilities')
@click.option('--use-cc', default=False, type=bool, help="Extract features using largest connected components.")
@click.option('--use-tracking-model', default=False, type=bool, help='Use an expectation-maximization style model to aid mouse tracking. Useful for data with cables')
@click.option('--tracking-model-ll-threshold', default=-100, type=float, help="Threshold on log-likelihood for pixels to use for update during tracking")
@click.option('--tracking-model-mask-threshold', default=-16, type=float, help="Threshold on log-likelihood to include pixels for centroid and angle calculation")
@click.option('--tracking-model-ll-clip', default=-100, type=float, help="Clip log-likelihoods below this value")
@click.option('--tracking-model-segment', default=True, type=bool, help="Segment likelihood mask from tracking model")
@click.option('--tracking-model-init', default='raw', type=str, help="Method for tracking model initialization")
@click.option('--tracking-init-mean', default=None, type=float, help="EM tracking initial mean.")
@click.option('--tracking-init-cov', default=None, type=float, help="EM tracking initial covariance.")
@click.option('--cable-filter-iters', default=0, type=int, help="Number of cable filter iterations")
@click.option('--cable-filter-shape', default='rectangle', type=str, help="Cable filter shape (rectangle or ellipse)")
@click.option('--cable-filter-size', default=(5, 5), type=(int, int), help="Cable filter size (in pixels)")
@click.option('--tail-filter-iters', default=1, type=int, help="Number of tail filter iterations")
@click.option('--tail-filter-size', default=(9, 9), type=(int, int), help='Tail filter size')
@click.option('--tail-filter-shape', default='ellipse', type=str, help='Tail filter shape')
@click.option('--spatial-filter-size', '-s', default=[3], type=int, help='Space prefilter kernel (median filter, must be odd)', multiple=True)
@click.option('--temporal-filter-size', '-t', default=[0], type=int, help='Time prefilter kernel (median filter, must be odd)', multiple=True)
@click.option('--chunk-size', default=1000, type=int, help='Number of frames for each processing iteration')
@click.option('--chunk-overlap', default=0, type=int, help='Frames overlapped in each chunk. Useful for cable tracking')
@click.option('--write-movie', default=True, type=bool, help='Write a results output movie including an extracted mouse')
@click.option('--frame-dtype', default='uint8', type=click.Choice(['uint8', 'uint16']), help='Data type for processed frames')
@click.option('--centroid-hampel-span', default=0, type=int, help='Hampel filter span')
@click.option('--centroid-hampel-sig', default=3, type=float, help='Hampel filter sig')
@click.option('--angle-hampel-span', default=0, type=int, help='Angle filter span')
@click.option('--angle-hampel-sig', default=3, type=float, help='Angle filter sig')
@click.option('--model-smoothing-clips', default=(0, 0), type=(float, float), help='Model smoothing clips')
@click.option('--frame-trim', default=(0, 0), type=(int, int), help='Frames to trim from beginning and end of data')
@click.option('--compress', default=False, type=bool, help='Convert .dat to .avi after successful extraction')
@click.option('--compress-chunk-size', type=int, default=3000, help='Chunk size for .avi compression')
@click.option('--compress-threads', type=int, default=3, help='Number of threads for encoding')
@click.option('--skip', is_flag=True, help='Will skip the extraction if it is already completed.')
def extract(input_file, crop_size, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, camera_type,
            bg_roi_depth_range, bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel,
            bg_roi_fill_holes, bg_sort_roi_by_position, bg_sort_roi_by_position_max_rois, dilate_iterations,
            min_height, max_height, detected_true_depth, fps, flip_classifier, flip_classifier_smoothing,
            use_tracking_model, tracking_model_ll_threshold, tracking_model_mask_threshold, use_cc,
            tracking_model_ll_clip, tracking_model_segment, tracking_model_init, cable_filter_iters, cable_filter_shape,
            cable_filter_size, tail_filter_iters, tail_filter_size, tail_filter_shape, spatial_filter_size,
            temporal_filter_size, chunk_size, chunk_overlap, output_dir, write_movie, use_plane_bground,
            frame_dtype, centroid_hampel_span, centroid_hampel_sig, angle_hampel_span, angle_hampel_sig,
            model_smoothing_clips, frame_trim, config_file, compress, compress_chunk_size, compress_threads,
            bg_roi_erode, erode_iterations, noise_tolerance, compute_raw_scalars, tracking_init_mean, tracking_init_cov,
            skip, progress_bar):


    click_data = click.get_current_context().params
    extract_wrapper(input_file, output_dir, click_data, skip=skip)

@cli.command(name="download-flip-file", help="Downloads Flip-correction model that helps with orienting the mouse during extraction.")
@click.argument('config-file', type=click.Path(exists=True, resolve_path=True), default='config.yaml')
@click.option('--output-dir', type=click.Path(),
              default=os.path.expanduser('~/moseq2'), help="Temp storage")
def download_flip_file(config_file, output_dir):

    flip_file_wrapper(config_file, output_dir)


@cli.command(name="generate-config", help="Generates a configuration file that holds editable options for extraction parameters.")
@click.option('--output-file', '-o', type=click.Path(), default='config.yaml')
def generate_config(output_file):

    objs = extract.params
    params = {tmp.name: tmp.default for tmp in objs if not tmp.required}

    with open(output_file, 'w') as f:
        yaml.safe_dump(params, f)

    print('Successfully generated config file in base directory.')

@cli.command(name='generate-index', help='Generates an index YAML file containing all extracted session metadata.')
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--pca-file', '-p', type=click.Path(), default=os.path.join(os.getcwd(), '_pca/pca_scores.h5'), help='Path to PCA results')
@click.option('--output-file', '-o', type=click.Path(), default=os.path.join(os.getcwd(), 'moseq2-index.yaml'), help="Location for storing index")
@click.option('--filter', '-f', type=(str, str), default=None, help='Regex filter for metadata', multiple=True)
@click.option('--all-uuids', '-a', type=bool, default=False, help='Use all uuids')
@click.option('--subpath', type=str, default='/proc/', help='Path substring to regulate paths included in an index file.')
def generate_index(input_dir, pca_file, output_file, filter, all_uuids, subpath):

    output_file = generate_index_wrapper(input_dir, pca_file, output_file, filter, all_uuids, subpath=subpath)

    if output_file != None:
        print(f'Index file: {output_file} was successfully generated.')

@cli.command(name='aggregate-results', help='Copies all extracted results (h5, yaml, avi) files from all extracted sessions to a new directory,')
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--format', '-f', type=str, default='{start_time}_{session_name}_{subject_name}', help='New file name formats from resepective metadata')
@click.option('--output-dir', '-o', type=click.Path(), default=os.path.join(os.getcwd(), 'aggregate_results/'), help="Location for storing all results together")
def aggregate_extract_results(input_dir, format, output_dir):

    aggregate_extract_results_wrapper(input_dir, format, output_dir)

@cli.command(name="convert-raw-to-avi", help='Converts/Compresses a raw depth file into an avi file (with depth values) that is 8x smaller.')
@click.argument('input-file', type=click.Path(exists=True, resolve_path=True))
@common_avi_options
def convert_raw_to_avi(input_file, output_file, chunk_size, fps, delete, threads):

    if output_file is None:
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(os.path.dirname(input_file), f'{base_filename}.avi')

    vid_info = get_movie_info(input_file)
    frame_batches = list(gen_batch_sequence(vid_info['nframes'], chunk_size, 0))
    video_pipe = None

    for batch in tqdm(frame_batches, desc='Encoding batches'):
        frames = load_movie_data(input_file, batch)
        video_pipe = write_frames(output_file,
                                  frames,
                                  pipe=video_pipe,
                                  close_pipe=False,
                                  threads=threads,
                                  fps=fps)

    if video_pipe:
        video_pipe.stdin.close()
        video_pipe.wait()

    for batch in tqdm(frame_batches, desc='Checking data integrity'):
        raw_frames = load_movie_data(input_file, batch)
        encoded_frames = load_movie_data(output_file, batch)

        if not np.array_equal(raw_frames, encoded_frames):
            raise RuntimeError(f'Raw frames and encoded frames not equal from {batch[0]} to {batch[-1]}')

    print('Encoding successful')

    if delete:
        print('Deleting', input_file)
        os.remove(input_file)


@cli.command(name="copy-slice", help='Copies a segment of an input depth recording into a new video file.')
@click.argument('input-file', type=click.Path(exists=True, resolve_path=True))
@common_avi_options
@click.option('-c', '--copy-slice', type=(int, int), default=(0, 1000), help='Slice to copy')
def copy_slice(input_file, output_file, copy_slice, chunk_size, fps, delete, threads):

    if output_file is None:
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        avi_encode = True
        output_file = os.path.join(os.path.dirname(input_file), f'{base_filename}.avi')
    else:
        output_filename, ext = os.path.splitext(os.path.basename(output_file))
        if ext == '.avi':
            avi_encode = True
        else:
            avi_encode = False

    vid_info = get_movie_info(input_file)
    copy_slice = (copy_slice[0], np.minimum(copy_slice[1], vid_info['nframes']))
    nframes = copy_slice[1] - copy_slice[0]
    offset = copy_slice[0]

    frame_batches = list(gen_batch_sequence(nframes, chunk_size, 0, offset))
    video_pipe = None

    if os.path.exists(output_file):
        overwrite = input('Press ENTER to overwrite your previous extraction, else to end the process.')
        if overwrite != '':
            sys.exit(0)

    for batch in tqdm(frame_batches, desc='Encoding batches'):
        frames = load_movie_data(input_file, batch)
        if avi_encode:
            video_pipe = write_frames(output_file,
                                      frames,
                                      pipe=video_pipe,
                                      close_pipe=False,
                                      threads=threads,
                                      fps=fps)
        else:
            with open(output_file, "ab") as f:
                f.write(frames.astype('uint16').tobytes())

    if avi_encode and video_pipe:
        video_pipe.stdin.close()
        video_pipe.wait()

    for batch in tqdm(frame_batches, desc='Checking data integrity'):
        raw_frames = load_movie_data(input_file, batch)
        encoded_frames = load_movie_data(output_file, batch)

        if not np.array_equal(raw_frames, encoded_frames):
            raise RuntimeError(f'Raw frames and encoded frames not equal from {batch[0]} to {batch[-1]}')

    print('Encoding successful')

    if delete:
        print('Deleting', input_file)
        os.remove(input_file)


if __name__ == '__main__':
    cli()