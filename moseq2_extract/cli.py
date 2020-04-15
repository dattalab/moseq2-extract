import os
import sys
import tqdm
import click
import pathlib
import numpy as np
import ruamel.yaml as yaml
from moseq2_extract.util import (gen_batch_sequence, command_with_config)
from moseq2_extract.io.video import (get_movie_info, load_movie_data, write_frames)
from moseq2_extract.helpers.wrappers import get_roi_wrapper, extract_wrapper, flip_file_wrapper

orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


@click.group()
def cli():
    pass


@cli.command(name="find-roi", cls=command_with_config('config_file'))
@click.argument('input-file', type=click.Path(exists=True))
@click.option('--bg-roi-dilate', default=(10, 10), type=(int, int), help='Size of strel to dilate roi')
@click.option('--bg-roi-shape', default='ellipse', type=str, help='Shape to use to dilate roi (ellipse or rect)')
@click.option('--bg-roi-index', default=[0], type=int, help='Index of roi to use', multiple=True)
@click.option('--bg-roi-weights', default=(1, .1, 1), type=(float, float, float), help='ROI feature weighting (area, extent, dist)')
@click.option('--bg-roi-depth-range', default=(650, 750), type=(float, float), help='Range to search for floor of arena (in mm)')
@click.option('--bg-roi-gradient-filter', default=False, type=bool, help='Exclude walls with gradient filtering')
@click.option('--bg-roi-gradient-threshold', default=3000, type=float, help='Gradient must be < this to include points')
@click.option('--bg-roi-gradient-kernel', default=7, type=int, help='Kernel size for Sobel gradient filtering')
@click.option('--bg-roi-fill-holes', default=True, type=bool, help='Fill holes in ROI')
@click.option('--bg-sort-roi-by-position', default=False, type=bool, help='Sort ROIs by position')
@click.option('--bg-sort-roi-by-position-max-rois', default=2, type=int, help='Max original ROIs to sort by position')
@click.option('--dilate_iterations', default=1, type=int, help='Number of dilation iterations to increase bucket floor size.')
@click.option('--output-dir', default=None, help='Output directory')
@click.option('--use-plane-bground', default=False, type=bool, help='Use plane fit for background')
@click.option("--config-file", type=click.Path())
def find_roi(input_file, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
             bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes,
             bg_sort_roi_by_position, bg_sort_roi_by_position_max_rois, dilate_iterations,
             output_dir, use_plane_bground, config_file):

    click_data = click.get_current_context().params
    cli_data = {k: v for k, v in click_data.items()}
    get_roi_wrapper(input_file, cli_data, output_dir)

@cli.command(name="extract", cls=command_with_config('config_file'))
@click.argument('input-file', type=click.Path(exists=True, resolve_path=True))
@click.option('--crop-size', '-c', default=(80, 80), type=(int, int), help='Width and height of cropped mouse image')
@click.option('--bg-roi-dilate', default=(10, 10), type=(int, int), help='Size of the mask dilation (to include environment walls)')
@click.option('--bg-roi-shape', default='ellipse', type=str, help='Shape to use for the mask dilation (ellipse or rect)')
@click.option('--bg-roi-index', default=0, type=int, help='Index of which background mask(s) to use')
@click.option('--bg-roi-weights', default=(1, .1, 1), type=(float, float, float), help='Feature weighting (area, extent, dist) of the background mask')
@click.option('--bg-roi-depth-range', default=(650, 750), type=(float, float), help='Range to search for floor of arena (in mm)')
@click.option('--bg-roi-gradient-filter', default=False, type=bool, help='Exclude walls with gradient filtering')
@click.option('--bg-roi-gradient-threshold', default=3000, type=float, help='Gradient must be < this to include points')
@click.option('--bg-roi-gradient-kernel', default=7, type=int, help='Kernel size for Sobel gradient filtering')
@click.option('--bg-roi-fill-holes', default=True, type=bool, help='Fill holes in ROI')
@click.option('--bg-sort-roi-by-position', default=False, type=bool, help='Sort ROIs by position')
@click.option('--bg-sort-roi-by-position-max-rois', default=2, type=int, help='Max original ROIs to sort by position')
@click.option('--dilate_iterations', default=1, type=int, help='Number of dilation iterations to increase bucket floor size.')
@click.option('--min-height', default=10, type=int, help='Min mouse height from floor (mm)')
@click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
@click.option('--detected-true-depth', default='auto', type=str, help='Option to override automatic depth estimation during extraction. Either "auto" or a int value.')
@click.option('--fps', default=30, type=int, help='Frame rate of camera')
@click.option('--flip-classifier', default=None, help='Location of the flip classifier used to properly orient the mouse (.pkl file)')
@click.option('--flip-classifier-smoothing', default=51, type=int, help='Number of frames to smooth flip classifier probabilities')
@click.option('--use-tracking-model', default=False, type=bool, help='Use an expectation-maximization style model to aid mouse tracking. Useful for data with cables')
@click.option('--tracking-model-ll-threshold', default=-100, type=float, help="Threshold on log-likelihood for pixels to use for update during tracking")
@click.option('--tracking-model-mask-threshold', default=-16, type=float, help="Threshold on log-likelihood to include pixels for centroid and angle calculation")
@click.option('--tracking-model-ll-clip', default=-100, type=float, help="Clip log-likelihoods below this value")
@click.option('--tracking-model-segment', default=True, type=bool, help="Segment likelihood mask from tracking model")
@click.option('--tracking-model-init', default='raw', type=str, help="Method for tracking model initialization")
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
@click.option('--output-dir', default=None, help='Output directory to save the results h5 file')
@click.option('--write-movie', default=True, type=bool, help='Write a results output movie including an extracted mouse')
@click.option('--use-plane-bground', is_flag=True, help='Use a plane fit for the background. Useful for mice that don\'t move much')
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
@click.option('--verbose', type=int, default=0, help='Level of verbosity during extraction process. [0-2]')
@click.option("--config-file", type=click.Path())
def extract(input_file, crop_size, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
            bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes,
            bg_sort_roi_by_position, bg_sort_roi_by_position_max_rois, dilate_iterations, min_height, max_height,
            detected_true_depth, fps, flip_classifier, flip_classifier_smoothing,
            use_tracking_model, tracking_model_ll_threshold, tracking_model_mask_threshold,
            tracking_model_ll_clip, tracking_model_segment, tracking_model_init, cable_filter_iters, cable_filter_shape,
            cable_filter_size, tail_filter_iters, tail_filter_size, tail_filter_shape, spatial_filter_size,
            temporal_filter_size, chunk_size, chunk_overlap, output_dir, write_movie, use_plane_bground,
            frame_dtype, centroid_hampel_span, centroid_hampel_sig, angle_hampel_span, angle_hampel_sig,
            model_smoothing_clips, frame_trim, config_file, compress, verbose, compress_chunk_size, compress_threads):

    click_data = click.get_current_context().params
    cli_data = {k: v for k, v in click_data.items()}
    extract_wrapper(input_file, output_dir, cli_data, extract=extract)



@cli.command(name="download-flip-file")
@click.argument('config-file', type=click.Path(exists=True, resolve_path=True), default='config.yaml')
@click.option('--output-dir', type=click.Path(),
              default=os.path.join(pathlib.Path.home(), 'moseq2'), help="Temp storage")
def download_flip_file(config_file, output_dir):

    flip_file_wrapper(config_file, output_dir)


@cli.command(name="generate-config")
@click.option('--output-file', '-o', type=click.Path(), default='config.yaml')
def generate_config(output_file):

    objs = extract.params
    params = {tmp.name: tmp.default for tmp in objs if not tmp.required}

    with open(output_file, 'w') as f:
        yaml.safe_dump(params, f)

    print('Successfully generated config file in base directory.')


@cli.command(name="convert-raw-to-avi")
@click.argument('input-file', type=click.Path(exists=True, resolve_path=True))
@click.option('-o', '--output-file', type=click.Path(), default=None, help='Path to output file')
@click.option('-b', '--chunk-size', type=int, default=3000, help='Chunk size')
@click.option('--fps', type=float, default=30, help='Video FPS')
@click.option('--delete', type=bool, is_flag=True, help='Delete raw file if encoding is successful')
@click.option('-t', '--threads', type=int, default=3, help='Number of threads for encoding')
@click.option('-v', '--verbose', type=int, default=0, help='Verbosity level out batch encoding. [0-1]')
def convert_raw_to_avi(input_file, output_file, chunk_size, fps, delete, threads, verbose):

    if output_file is None:
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(os.path.dirname(input_file), f'{base_filename}.avi')

    vid_info = get_movie_info(input_file)
    frame_batches = list(gen_batch_sequence(vid_info['nframes'], chunk_size, 0))
    video_pipe = None

    for batch in tqdm.tqdm(frame_batches, desc='Encoding batches'):
        frames = load_movie_data(input_file, batch)
        video_pipe = write_frames(output_file,
                                  frames,
                                  pipe=video_pipe,
                                  close_pipe=False,
                                  threads=threads,
                                  fps=fps, verbose=verbose)

    if video_pipe:
        video_pipe.stdin.close()
        video_pipe.wait()

    for batch in tqdm.tqdm(frame_batches, desc='Checking data integrity'):
        raw_frames = load_movie_data(input_file, batch)
        encoded_frames = load_movie_data(output_file, batch)

        if not np.array_equal(raw_frames, encoded_frames):
            raise RuntimeError(f'Raw frames and encoded frames not equal from {batch[0]} to {batch[-1]}')

    print('Encoding successful')

    if delete:
        print('Deleting', input_file)
        os.remove(input_file)


@cli.command(name="copy-slice")
@click.argument('input-file', type=click.Path(exists=True, resolve_path=True))
@click.option('-o', '--output-file', type=click.Path(), default=None, help='Path to output file')
@click.option('-b', '--chunk-size', type=int, default=3000, help='Chunk size')
@click.option('-c', '--copy-slice', type=(int, int), default=(0, 1000), help='Slice to copy')
@click.option('--fps', type=float, default=30, help='Video FPS')
@click.option('--delete', type=bool, is_flag=True, help='Delete raw file if encoding is sucessful')
@click.option('-t', '--threads', type=int, default=3, help='Number of threads for encoding')
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

    for batch in tqdm.tqdm(frame_batches, desc='Encoding batches'):
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

    for batch in tqdm.tqdm(frame_batches, desc='Checking data integrity'):
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
