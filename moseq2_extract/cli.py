'''
CLI front-end operations. This module contains all the functionality and configurable parameters
users can alter to most accurately process their data.

Note: These functions simply read all the parameters into a dictionary,
 and then call the corresponding wrapper function with the given input parameters.
'''

import os
import click
import ruamel.yaml as yaml
from tqdm.auto import tqdm
from copy import deepcopy
from moseq2_extract.util import command_with_config, read_yaml, recursive_find_unextracted_dirs
from moseq2_extract.helpers.wrappers import (get_roi_wrapper, extract_wrapper, flip_file_wrapper,
                                             generate_index_wrapper, aggregate_extract_results_wrapper,
                                             convert_raw_to_avi_wrapper, copy_slice_wrapper)

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
    function = click.option('--dilate-iterations', default=1, type=int,
                            help='Number of dilation iterations to increase bucket floor size.')(function)
    function = click.option('--bg-roi-erode', default=(1, 1), type=(int, int),
                            help='Size of cv2 Structure Element to erode roi. (Special Cases Only)')(function)
    function = click.option('--erode-iterations', default=0, type=int,
                            help='Number of erosion iterations to decrease bucket floor size. (Special Cases Only)')(function)
    function = click.option('--noise-tolerance', default=30, type=int,
                            help='Extent of noise to accept during RANSAC Plane ROI computation. (Special Cases Only)')(function)
    function = click.option('--output-dir', default='proc', help='Output directory to save the results h5 file')(function)
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

def extract_options(function):
    function = click.option('--crop-size', '-c', default=(80, 80), type=(int, int), help='Width and height of cropped mouse image')(function)
    function = click.option('--num-frames', '-n', default=None, type=int, help='Number of frames to extract. Will extract full session if set to None.')(function)
    function = click.option('--min-height', default=10, type=int, help='Min mouse height from floor (mm)')(function)
    function = click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')(function)
    function = click.option('--detected-true-depth', default='auto', type=str, help='Option to override automatic depth estimation during extraction. \
This is only a debugging parameter, for cases where dilate_iterations > 1, otherwise has no effect. Either "auto" or an int value.')(function)
    function = click.option('--compute-raw-scalars', is_flag=True, help="Compute scalar values from raw cropped frames.")(function)
    function = click.option('--flip-classifier', default=None, help='Location of the flip classifier used to properly orient the mouse (.pkl file)')(function)
    function = click.option('--flip-classifier-smoothing', default=51, type=int, help='Number of frames to smooth flip classifier probabilities')(function)
    function = click.option('--graduate-walls', default=False, type=bool, help="Graduates and dilates the background image to compensate for slanted bucket walls. \\_/")(function)
    function = click.option('--widen-radius', default=0, type=int, help="Number of pixels to increase/decrease radius by when graduating bucket walls.")(function)
    function = click.option('--use-cc', default=True, type=bool, help="Extract features using largest connected components.")(function)
    function = click.option('--use-tracking-model', default=False, type=bool, help='Use an expectation-maximization style model to aid mouse tracking. Useful for data with cables')(function)
    function = click.option('--tracking-model-ll-threshold', default=-100, type=float, help="Threshold on log-likelihood for pixels to use for update during tracking")(function)
    function = click.option('--tracking-model-mask-threshold', default=-16, type=float, help="Threshold on log-likelihood to include pixels for centroid and angle calculation")(function)
    function = click.option('--tracking-model-ll-clip', default=-100, type=float, help="Clip log-likelihoods below this value")(function)
    function = click.option('--tracking-model-segment', default=True, type=bool, help="Segment likelihood mask from tracking model")(function)
    function = click.option('--tracking-model-init', default='raw', type=str, help="Method for tracking model initialization")(function)
    function = click.option('--cable-filter-iters', default=0, type=int, help="Number of cable filter iterations")(function)
    function = click.option('--cable-filter-shape', default='rectangle', type=str, help="Cable filter shape (rectangle or ellipse)")(function)
    function = click.option('--cable-filter-size', default=(5, 5), type=(int, int), help="Cable filter size (in pixels)")(function)
    function = click.option('--tail-filter-iters', default=1, type=int, help="Number of tail filter iterations")(function)
    function = click.option('--tail-filter-size', default=(9, 9), type=(int, int), help='Tail filter size')(function)
    function = click.option('--tail-filter-shape', default='ellipse', type=str, help='Tail filter shape')(function)
    function = click.option('--spatial-filter-size', '-s', default=[3], type=int, help='Space prefilter kernel (median filter, must be odd)', multiple=True)(function)
    function = click.option('--temporal-filter-size', '-t', default=[0], type=int, help='Time prefilter kernel (median filter, must be odd)', multiple=True)(function)
    function = click.option('--chunk-overlap', default=0, type=int, help='Frames overlapped in each chunk. Useful for cable tracking')(function)
    function = click.option('--write-movie', default=True, type=bool, help='Write a results output movie including an extracted mouse')(function)
    function = click.option('--frame-dtype', default='uint8', type=click.Choice(['uint8', 'uint16']), help='Data type for processed frames')(function)
    function = click.option('--centroid-hampel-span', default=0, type=int, help='Hampel filter span')(function)
    function = click.option('--centroid-hampel-sig', default=3, type=float, help='Hampel filter sig')(function)
    function = click.option('--angle-hampel-span', default=0, type=int, help='Angle filter span')(function)
    function = click.option('--angle-hampel-sig', default=3, type=float, help='Angle filter sig')(function)
    function = click.option('--model-smoothing-clips', default=(0, 0), type=(float, float), help='Model smoothing clips')(function)
    function = click.option('--frame-trim', default=(0, 0), type=(int, int), help='Frames to trim from beginning and end of data')(function)
    function = click.option('--compress', default=False, type=bool, help='Convert .dat to .avi after successful extraction')(function)
    function = click.option('--compress-chunk-size', type=int, default=3000, help='Chunk size for .avi compression')(function)
    function = click.option('--compress-threads', type=int, default=3, help='Number of threads for encoding')(function)
    function = click.option('--skip-completed', is_flag=True, help='Will skip the extraction if it is already completed.')(function)

    return function

@cli.command(name="find-roi", cls=command_with_config('config_file'), help="Finds the ROI and background distance to subtract from frames when extracting.")
@click.argument('input-file', type=click.Path(exists=True))
@common_roi_options
def find_roi(input_file, output_dir, **config_data):

    get_roi_wrapper(input_file, config_data, output_dir)

@cli.command(name="extract", cls=command_with_config('config_file'),
             help="Processes raw input depth recordings to output a cropped and oriented"
             "video of the mouse and saves the output+metadata to h5 files in the given output directory.")
@click.argument('input-file', type=click.Path(exists=True, resolve_path=True))
@common_roi_options
@common_avi_options
@extract_options
def extract(input_file, output_dir, num_frames, skip_completed, **config_data):

    extract_wrapper(input_file, output_dir, config_data, num_frames=num_frames, skip=skip_completed)


@cli.command(name='batch-extract', cls=command_with_config('config_file'), help='Batch processes '
             'all the raw depth recordings located in the input folder.')
@click.argument('input-folder', type=click.Path(exists=True, resolve_path=True))
@common_roi_options
@common_avi_options
@extract_options
@click.option('--extensions', default=['.dat'], type=str, help='File extension of raw data', multiple=True)
@click.option('--skip-checks', is_flag=True, help='Flag: skip checks for the existance of a metadata file')
def batch_extract(input_folder, output_dir, skip_completed, num_frames, extensions,
                  skip_checks, **config_data):

    to_extract = []
    for ex in extensions:
        to_extract.extend(
            recursive_find_unextracted_dirs(input_folder, extension=ex,
                skip_checks=True if ex in ('.tgz', '.tar.gz') else skip_checks,
                 yaml_path=os.path.join(output_dir, 'results_00.yaml')))
    for session in tqdm(to_extract):
        extract_wrapper(session, output_dir, deepcopy(config_data), num_frames=num_frames,
                        skip=skip_completed)

    

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
@click.option('--output-file', '-o', type=click.Path(), default=os.path.join(os.getcwd(), 'moseq2-index.yaml'), help="Location for storing index")
def generate_index(input_dir, output_file):

    output_file = generate_index_wrapper(input_dir, output_file)

    if output_file is not None:
        print(f'Index file: {output_file} was successfully generated.')

@cli.command(name='aggregate-results', help='Copies all extracted results (h5, yaml, avi) files from all extracted sessions to a new directory,')
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--format', '-f', type=str, default='{start_time}_{session_name}_{subject_name}', help='New file name formats from resepective metadata')
@click.option('--output-dir', '-o', type=click.Path(), default=os.path.join(os.getcwd(), 'aggregate_results/'), help="Location for storing all results together")
@click.option('--mouse-threshold', default=0, type=float, help='Threshold value for mean depth to include frames in aggregated results')
def aggregate_extract_results(input_dir, format, output_dir, mouse_threshold):

    aggregate_extract_results_wrapper(input_dir, format, output_dir, mouse_threshold)

@cli.command(name="convert-raw-to-avi", help='Converts/Compresses a raw depth file into an avi file (with depth values) that is 8x smaller.')
@click.argument('input-file', type=click.Path(exists=True, resolve_path=True))
@common_avi_options
def convert_raw_to_avi(input_file, output_file, chunk_size, fps, delete, threads):

    convert_raw_to_avi_wrapper(input_file, output_file, chunk_size, fps, delete, threads)

@cli.command(name="copy-slice", help='Copies a segment of an input depth recording into a new video file.')
@click.argument('input-file', type=click.Path(exists=True, resolve_path=True))
@common_avi_options
@click.option('-c', '--copy-slice', type=(int, int), default=(0, 1000), help='Slice indices used for copy')
def copy_slice(input_file, output_file, copy_slice, chunk_size, fps, delete, threads):

    copy_slice_wrapper(input_file, output_file, copy_slice, chunk_size, fps, delete, threads)

if __name__ == '__main__':
    cli()