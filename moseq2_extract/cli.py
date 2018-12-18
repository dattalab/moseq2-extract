from moseq2_extract.io.video import (get_movie_info, load_movie_data,
                                     write_frames_preview, write_frames)
from moseq2_extract.io.image import write_image, read_image
from moseq2_extract.extract.extract import extract_chunk
from moseq2_extract.extract.proc import apply_roi, get_roi, get_bground_im_file
from moseq2_extract.util import (load_metadata, gen_batch_sequence, load_timestamps,
                                 select_strel, command_with_config, scalar_attributes, 
                                 save_dict_contents_to_h5)
import click
import os
import h5py
import tqdm
import numpy as np
import ruamel.yaml as yaml
import uuid
import pathlib
import urllib.request
from copy import deepcopy


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
@click.option('--output-dir', default=None, help='Output directory')
@click.option('--use-plane-bground', default=False, type=bool, help='Use plane fit for background')
@click.option("--config-file", type=click.Path())
def find_roi(input_file, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
             bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel,
             output_dir, use_plane_bground, config_file):

    # set up the output directory

    if type(bg_roi_index) is int:
        bg_roi_index = [bg_roi_index]

    if not output_dir:
        output_dir = os.path.join(os.path.dirname(input_file), 'proc')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(os.path.join(output_dir, 'bground.tiff')):
        print('Loading background...')
        bground_im = read_image(os.path.join(output_dir, 'bground.tiff'), scale=True)
    else:
        print('Getting background...')
        bground_im = get_bground_im_file(input_file)
        write_image(os.path.join(output_dir, 'bground.tiff'), bground_im, scale=True)

    first_frame = load_movie_data(input_file, 0)
    write_image(os.path.join(output_dir, 'first_frame.tiff'), first_frame, scale=True,
                scale_factor=bg_roi_depth_range)

    print('Getting roi...')
    strel_dilate = select_strel(bg_roi_shape, bg_roi_dilate)

    rois, _, _, _, _, _ = get_roi(bground_im,
                                  strel_dilate=strel_dilate,
                                  weights=bg_roi_weights,
                                  depth_range=bg_roi_depth_range,
                                  gradient_filter=bg_roi_gradient_filter,
                                  gradient_threshold=bg_roi_gradient_threshold,
                                  gradient_kernel=bg_roi_gradient_kernel)

    bg_roi_index = [idx for idx in bg_roi_index if idx in range(len(rois))]
    for idx in bg_roi_index:
        roi_filename = 'roi_{:02d}.tiff'.format(idx)
        write_image(os.path.join(output_dir, roi_filename),
                    rois[idx], scale=True, dtype='uint8')


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
@click.option('--min-height', default=10, type=int, help='Min mouse height from floor (mm)')
@click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
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
@click.option("--config-file", type=click.Path())
def extract(input_file, crop_size, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
            bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel,
            min_height, max_height, fps, flip_classifier, flip_classifier_smoothing,
            use_tracking_model, tracking_model_ll_threshold, tracking_model_mask_threshold,
            tracking_model_ll_clip, tracking_model_segment, tracking_model_init, cable_filter_iters, cable_filter_shape,
            cable_filter_size, tail_filter_iters, tail_filter_size, tail_filter_shape, spatial_filter_size,
            temporal_filter_size, chunk_size, chunk_overlap, output_dir, write_movie, use_plane_bground,
            frame_dtype, centroid_hampel_span, centroid_hampel_sig, angle_hampel_span, angle_hampel_sig,
            model_smoothing_clips, frame_trim, config_file):

    print('Processing: {}'.format(input_file))
    # get the basic metadata

    # if we pass in multiple roi indices, recurse and process each roi
    # if len(bg_roi_index) > 1:
    #     for roi in bg_roi_index:
    #         extract(bg_roi_index=roi, **locals())
    #     return None

    status_dict = {
        'parameters': deepcopy(locals()),
        'complete': False,
        'skip': False,
        'uuid': str(uuid.uuid4()),
        'metadata': ''
    }

    # np.seterr(invalid='raise')

    video_metadata = get_movie_info(input_file)
    nframes = video_metadata['nframes']

    if frame_trim[0] and frame_trim[0] < nframes:
        first_frame_idx = frame_trim[0]
    else:
        first_frame_idx = 0

    if nframes - frame_trim[1] > first_frame_idx:
        last_frame_idx = nframes - frame_trim[1]
    else:
        last_frame_idx = nframes

    nframes = last_frame_idx - first_frame_idx

    metadata_path = os.path.join(os.path.dirname(input_file), 'metadata.json')
    timestamp_path = os.path.join(os.path.dirname(input_file), 'depth_ts.txt')

    if os.path.exists(metadata_path):
        acquisition_metadata = load_metadata(metadata_path)
        status_dict['metadata'] = acquisition_metadata
    else:
        acquisition_metadata = {}

    if os.path.exists(timestamp_path):
        timestamps = load_timestamps(timestamp_path, col=0)[first_frame_idx:last_frame_idx]
    else:
        timestamps = None

    scalars_attrs = scalar_attributes()
    scalars = list(scalars_attrs.keys())

    frame_batches = list(gen_batch_sequence(nframes, chunk_size, chunk_overlap))

    # set up the output directory

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_file), 'proc')
    else:
        output_dir = os.path.join(os.path.dirname(input_file), output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = 'results_{:02d}'.format(bg_roi_index)
    status_filename = os.path.join(output_dir, '{}.yaml'.format(output_filename))

    if os.path.exists(status_filename):
        raise RuntimeError("Already found a status file in {}, delete and try again".format(status_filename))

    with open(status_filename, 'w') as f:
        yaml.dump(status_dict, f, Dumper=yaml.RoundTripDumper)

    # get the background and roi, which will be used across all batches

    if os.path.exists(os.path.join(output_dir, 'bground.tiff')):
        print('Loading background...')
        bground_im = read_image(os.path.join(output_dir, 'bground.tiff'), scale=True)
    else:
        print('Getting background...')
        bground_im = get_bground_im_file(input_file)
        if not use_plane_bground:
            write_image(os.path.join(output_dir, 'bground.tiff'), bground_im, scale=True)

    first_frame = load_movie_data(input_file, 0)
    write_image(os.path.join(output_dir, 'first_frame.tiff'), first_frame, scale=True,
                scale_factor=bg_roi_depth_range)

    roi_filename = 'roi_{:02d}.tiff'.format(bg_roi_index)

    strel_dilate = select_strel(bg_roi_shape, bg_roi_dilate)
    strel_tail = select_strel(tail_filter_shape, tail_filter_size)
    strel_min = select_strel(cable_filter_shape, cable_filter_size)

    if os.path.exists(os.path.join(output_dir, roi_filename)):
        print('Loading ROI...')
        roi = read_image(os.path.join(output_dir, roi_filename), scale=True) > 0
    else:
        print('Getting roi...')
        rois, plane, _, _, _, _ = get_roi(bground_im,
                                          strel_dilate=strel_dilate,
                                          weights=bg_roi_weights,
                                          depth_range=bg_roi_depth_range,
                                          gradient_filter=bg_roi_gradient_filter,
                                          gradient_threshold=bg_roi_gradient_threshold,
                                          gradient_kernel=bg_roi_gradient_kernel)

        if use_plane_bground:
            print('Using plane fit for background...')
            xx, yy = np.meshgrid(np.arange(bground_im.shape[1]), np.arange(bground_im.shape[0]))
            coords = np.vstack((xx.ravel(), yy.ravel()))
            plane_im = (np.dot(coords.T, plane[:2]) + plane[3]) / -plane[2]
            plane_im = plane_im.reshape(bground_im.shape)
            write_image(os.path.join(output_dir, 'bground.tiff'), plane_im, scale=True)
            bground_im = plane_im

        roi = rois[bg_roi_index]
        write_image(os.path.join(output_dir, roi_filename),
                    roi, scale=True, dtype='uint8')

    true_depth = np.median(bground_im[roi > 0])
    print('Detected true depth: {}'.format(true_depth))

    # farm out the batches and write to an hdf5 file

    with h5py.File(os.path.join(output_dir, '{}.h5'.format(output_filename)), 'w') as f:
        f.create_dataset('metadata/uuid', data=status_dict['uuid'])
        for scalar in scalars:
            f.create_dataset('scalars/{}'.format(scalar), (nframes,), 'float32', compression='gzip')
            f['scalars/{}'.format(scalar)].attrs['description'] = scalars_attrs[scalar]

        if timestamps is not None:
            f.create_dataset('timestamps', compression='gzip', data=timestamps)
        f.create_dataset('frames', (nframes, crop_size[0], crop_size[1]), frame_dtype, compression='gzip')
        f['frames'].attrs['description'] = '3D Numpy array of depth frames (nframes x w x h, in mm)'

        if use_tracking_model:
            f.create_dataset('frames_mask', (nframes, crop_size[0], crop_size[1]), 'float32', compression='gzip')
            f['frames_mask'].attrs['description'] = 'Log-likelihood values from the tracking model (nframes x w x h)'
        else:
            f.create_dataset('frames_mask', (nframes, crop_size[0], crop_size[1]), 'bool', compression='gzip')
            f['frames_mask'].attrs['description'] = 'Boolean mask, false=not mouse, true=mouse'

        if flip_classifier is not None:
            f.create_dataset('metadata/extraction/flips', (nframes, ), 'bool', compression='gzip')
            f['metadata/extraction/flips'].attrs['description'] = 'Output from flip classifier, false=no flip, true=flip'

        # if use_tracking_model:
        #     f.create_dataset('frames_ll', (nframes, crop_size[0], crop_size[1]),
        #                      'float32', compression='gzip')
        
        f.create_dataset('metadata/extraction/true_depth', data=true_depth)
        f['metadata/extraction/true_depth'].attrs['description'] = 'Detected true depth of arena floor in mm'

        f.create_dataset('metadata/extraction/roi', data=roi, compression='gzip')
        f['metadata/extraction/roi'].attrs['description'] = 'ROI mask'

        f.create_dataset('metadata/extraction/first_frame', data=first_frame[0], compression='gzip')
        f['metadata/extraction/first_frame'].attrs['description'] = 'First frame of depth dataset'

        f.create_dataset('metadata/extraction/background', data=bground_im, compression='gzip')
        f['metadata/extraction/background'].attrs['description'] = 'Computed background image'

        save_dict_contents_to_h5(f, status_dict['parameters'], 'metadata/extraction/parameters')
        
        for key, value in acquisition_metadata.items():

            if type(value) is list and len(value) > 0 and type(value[0]) is str:
                value = [n.encode('utf8') for n in value]

            f.create_dataset('metadata/acquisition/{}'.format(key), data=value)

        video_pipe = None
        tracking_init_mean = None
        tracking_init_cov = None

        for i, frame_range in enumerate(tqdm.tqdm(frame_batches, desc='Processing batches')):
            raw_frames = load_movie_data(input_file, [f + first_frame_idx for f in frame_range])
            raw_frames = bground_im-raw_frames
            # raw_frames[np.logical_or(raw_frames < min_height, raw_frames > max_height)] = 0
            raw_frames[raw_frames < min_height] = 0
            raw_frames[raw_frames > max_height] = max_height
            raw_frames = raw_frames.astype(frame_dtype)
            raw_frames = apply_roi(raw_frames, roi)

            results = extract_chunk(raw_frames,
                                    use_em_tracker=use_tracking_model,
                                    strel_tail=strel_tail,
                                    strel_min=strel_min,
                                    iters_tail=tail_filter_iters,
                                    iters_min=cable_filter_iters,
                                    prefilter_space=spatial_filter_size,
                                    prefilter_time=temporal_filter_size,
                                    min_height=min_height,
                                    max_height=max_height,
                                    flip_classifier=flip_classifier,
                                    flip_smoothing=flip_classifier_smoothing,
                                    crop_size=crop_size,
                                    frame_dtype=frame_dtype,
                                    mask_threshold=tracking_model_mask_threshold,
                                    tracking_ll_threshold=tracking_model_ll_threshold,
                                    tracking_segment=tracking_model_segment,
                                    tracking_init_mean=tracking_init_mean,
                                    tracking_init_cov=tracking_init_cov,
                                    true_depth=true_depth,
                                    centroid_hampel_span=centroid_hampel_span,
                                    centroid_hampel_sig=centroid_hampel_sig,
                                    angle_hampel_span=angle_hampel_span,
                                    angle_hampel_sig=angle_hampel_sig,
                                    model_smoothing_clips=model_smoothing_clips,
                                    tracking_model_init=tracking_model_init)

            # if desired, write out a movie

            if i > 0:
                offset = chunk_overlap
            else:
                offset = 0

            if use_tracking_model:
                results['mask_frames'][results['depth_frames'] < min_height] = tracking_model_ll_clip
                results['mask_frames'][results['mask_frames'] < tracking_model_ll_clip] = tracking_model_ll_clip
                tracking_init_mean = results['parameters']['mean'][-(chunk_overlap+1)]
                tracking_init_cov = results['parameters']['cov'][-(chunk_overlap+1)]

            frame_range = frame_range[offset:]

            for scalar in scalars:
                f['scalars/{}'.format(scalar)][frame_range] = results['scalars'][scalar][offset:, ...]

            f['frames'][frame_range] = results['depth_frames'][offset:, ...]
            f['frames_mask'][frame_range] = results['mask_frames'][offset:, ...]

            if flip_classifier:
                f['metadata/extraction/flips'][frame_range] = results['flips'][offset:]

            nframes, rows, cols = raw_frames[offset:, ...].shape
            output_movie = np.zeros((nframes, rows+crop_size[0], cols+crop_size[1]), 'uint16')
            output_movie[:, :crop_size[0], :crop_size[1]] = results['depth_frames'][offset:, ...]
            output_movie[:, crop_size[0]:, crop_size[1]:] = raw_frames[offset:, ...]

            video_pipe = write_frames_preview(
                os.path.join(output_dir, '{}.mp4'.format(output_filename)), output_movie,
                pipe=video_pipe, close_pipe=False, fps=fps, frame_range=[f + first_frame_idx for f in frame_range])

        if video_pipe:
            video_pipe.stdin.close()
            video_pipe.wait()

    status_dict['complete'] = True

    with open(status_filename, 'w') as f:
        yaml.dump(status_dict, f, Dumper=yaml.RoundTripDumper)

    print('\n')


@cli.command(name="download-flip-file")
@click.option('--output-dir', type=click.Path(),
              default=os.path.join(pathlib.Path.home(), 'moseq2'), help="Temp storage")
def download_flip_file(output_dir):

    # TODO: more flip files!!!!
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
        print('[{}] {} ---> {}'.format(idx, k, v))

    selection = None

    while selection is None:
        selection = click.prompt('Enter a selection', type=int)
        if selection > len(flip_files.keys()):
            selection = None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    selection = flip_files[key_list[selection]]

    output_filename = os.path.join(output_dir, os.path.basename(selection))
    urllib.request.urlretrieve(selection, output_filename)
    print('Successfully downloaded flip file to {}'.format(output_filename))
    print('Be sure to supply this as your flip-file during extraction')


@cli.command(name="generate-config")
@click.option('--output-file', '-o', type=click.Path(), default='config.yaml')
def generate_config(output_file):
    objs = extract.params
    params = {tmp.name: tmp.default for tmp in objs if not tmp.required}

    with open(output_file, 'w') as f:
        yaml.dump(params, f, Dumper=yaml.RoundTripDumper)


@cli.command(name="convert-raw-to-avi")
@click.argument('input-file', type=click.Path(exists=True, resolve_path=True))
@click.option('-o', '--output-file', type=click.Path(), default=None, help='Path to output file')
@click.option('-b', '--chunk-size', type=int, default=3000, help='Chunk size')
@click.option('--fps', type=float, default=30, help='Video FPS')
@click.option('--delete', type=bool, is_flag=True, help='Delete raw file if encoding is sucessful')
@click.option('-t', '--threads', type=int, default=3, help='Number of threads for encoding')
def convert_raw_to_avi(input_file, output_file, chunk_size, fps, delete, threads):

    if output_file is None:
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(os.path.dirname(input_file),
                                   '{}.avi'.format(base_filename))

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
                                  fps=fps)

    if video_pipe:
        video_pipe.stdin.close()
        video_pipe.wait()

    for batch in tqdm.tqdm(frame_batches, desc='Checking data integrity'):
        raw_frames = load_movie_data(input_file, batch)
        encoded_frames = load_movie_data(output_file, batch)

        if not np.array_equal(raw_frames, encoded_frames):
            raise RuntimeError('Raw frames and encoded frames not equal from {} to {}'.format(batch[0], batch[-1]))

    print('Encoding successful')

    if delete:
        print('Deleting {}'.format(input_file))
        os.remove(input_file)


if __name__ == '__main__':
    cli()
