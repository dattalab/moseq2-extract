import os
import h5py
import click
import tqdm
import numpy as np
import moseq2.config as config
from moseq2.io.video import get_movie_info,\
    load_movie_data, write_frames_preview
from moseq2.io.image import write_image, read_image
from moseq2.extract.extract import extract_chunk
from moseq2.extract.proc import apply_roi, get_roi, get_bground_im_file
from moseq2.util import load_metadata, gen_batch_sequence, load_timestamps, select_strel


@click.group()
def cli():
    pass


@cli.command(name="find-roi")
@click.argument('input-file', type=click.Path(exists=True))
@click.option('--roi-dilate', default=(10, 10), type=tuple, help='Size of strel to dilate roi')
@click.option('--roi-shape', default='ellipse', type=str, help='Shape to use to dilate roi (ellipse or rect)')
@click.option('--roi-index', default=0, type=int, help='Index of roi to use')
@click.option('--roi-weights', default=(1, .1, 1), type=(float, float, float),
              help='ROI feature weighting (area, extent, dist)')
@click.option('--output-dir', default=None, help='Output directory')
@click.option('--use-plane-bground', default=False, type=bool, help='Use plane fit for background')
def find_roi(input_file, roi_dilate, roi_shape, roi_index, roi_weights, output_dir, use_plane_bground):

    # set up the output directory

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
                scale_factor=(650, 750))

    print('Getting roi...')
    strel_dilate = select_strel(roi_shape, roi_dilate)

    roi_filename = 'roi_{:02d}.tiff'.format(roi_index)
    rois, _, _, _, _, _ = get_roi(bground_im, strel_dilate=strel_dilate, weights=roi_weights)
    write_image(os.path.join(output_dir, roi_filename),
                rois[roi_index], scale=True, dtype='uint8')


@cli.command(name="extract")
@click.argument('input-file', type=click.Path(exists=True))
@click.option('--crop-size', '-c', default=None, type=tuple, help='Width and height of cropped mouse')
@click.option('--roi-dilate', default=(), type=tuple, help='Size of strel to dilate roi')
@click.option('--roi-shape', default=None, type=str, help='Shape to use to dilate roi (ellipse or rect)')
@click.option('--roi-index', default=None, type=int, help='Index of roi to use')
@click.option('--roi-weights', default=(), type=tuple,
              help='ROI feature weighting (area, extent, dist)')
@click.option('--min-height', default=None, type=int, help='Min height of mouse from floor (mm)')
@click.option('--max-height', default=None, type=int, help='Max height of mouse from floor (mm)')
@click.option('--fps', default=None, type=int, help='Frame rate of camera')
@click.option('--flip-file', default=None, help='Location of flip classifier (.pkl)')
@click.option('--em-tracking', is_flag=True, help='Use em tracker')
@click.option('--prefilter-space', default=(), type=tuple, help='Space prefilter kernel')
@click.option('--prefilter-time', default=(), type=tuple, help='Time prefilter kernel')
@click.option('--chunk-size', default=None, type=int, help='Chunk size for processing')
@click.option('--chunk-overlap', default=None, type=int, help='Overlap in chunks')
@click.option('--output-dir', default=None, help='Output directory')
@click.option('--write-movie', default=None, type=bool, help='Write results movie')
@click.option('--use-plane-bground', is_flag=True, help='Use plane fit for background')
@click.option('--config', default=None, type=click.Path(exists=True, resolve_path=True),
              help='Path to a moseq config file describing the parameters used for extraction')
def extract(**cliargs):

    np.seterr(invalid='raise')

    # first deal with config file here
    if cliargs['config'] is not None:
        conf = config.load_config(cliargs['config'])
    else:
        # get default params
        conf = config.create_config()

    # merge defaults and config choices
    args = config.merge_cli_config(cliargs, conf)

    # get the basic metadata
    video_metadata = get_movie_info(args['input_file'])
    nframes = video_metadata['nframes']
    extraction_metadata = load_metadata(os.path.join(os.path.dirname(args['input_file']), 'metadata.json'))
    timestamps = load_timestamps(os.path.join(os.path.dirname(args['input_file']), 'depth_ts.txt'), col=0)
    frame_batches = list(gen_batch_sequence(nframes, args['chunk_size'], args['chunk_overlap']))

    # set up the output directory
    output_dir = os.path.join(os.path.dirname(args['input_file']), args['output_dir'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare an hdf5 file for all the resulting output,
    # dump videos in a temp directory to be stitched later

    # get the background and roi, which will be used across all batches

    if os.path.exists(os.path.join(output_dir, 'bground.tiff')):
        print('Loading background...')
        bground_im = read_image(os.path.join(output_dir, 'bground.tiff'), scale=True)
    else:
        print('Getting background...')
        bground_im = get_bground_im_file(args['input_file'])
        if not args['use_plane_bground']:
            write_image(os.path.join(output_dir, 'bground.tiff'), bground_im, scale=True)

    first_frame = load_movie_data(args['input_file'], frames=0)
    write_image(os.path.join(output_dir, 'first_frame.tiff'), first_frame, scale=True,
                scale_factor=(650, 750))

    roi_filename = 'roi_{:02d}.tiff'.format(args['roi_index'])

    if os.path.exists(os.path.join(output_dir, roi_filename)):
        print('Loading ROI...')
        roi = read_image(os.path.join(output_dir, roi_filename), scale=True) > 0
    else:
        print('Getting roi...')
        strel_dilate = select_strel(args['roi_shape'], args['roi_dilate'])
        rois, plane, _, _, _, _ = get_roi(bground_im, strel_dilate=strel_dilate, weights=args['roi_weights'])

        if args['use_plane_bground']:
            print('Using plane fit for background...')
            xx, yy = np.meshgrid(np.arange(bground_im.shape[1]), np.arange(bground_im.shape[0]))
            coords = np.vstack((xx.ravel(), yy.ravel()))
            plane_im = (np.dot(coords.T, plane[:2]) + plane[3]) / -plane[2]
            plane_im = plane_im.reshape(bground_im.shape)
            write_image(os.path.join(output_dir, 'bground.tiff'), plane_im, scale=True)
            bground_im = plane_im

        roi = rois[args['roi_index']]
        write_image(os.path.join(output_dir, roi_filename),
                    roi, scale=True, dtype='uint8')

    # define scalars right before first use
    scalars = ['centroid_x', 'centroid_y', 'angle', 'width',
               'length', 'height_ave', 'velocity_mag',
               'velocity_theta', 'area', 'velocity_mag_3d']

    # farm out the batches and write to an hdf5 file
    with h5py.File(os.path.join(output_dir, 'results.h5'), 'w') as f:
        for i in range(len(scalars)):
            f.create_dataset('scalars/{}'.format(scalars[i]), (nframes,), 'float32', compression='gzip')

        f.create_dataset('metadata/timestamps', compression='gzip', data=timestamps)
        f.create_dataset('frames', (nframes, args['crop_size'][0], args['crop_size'][1]), 'i1', compression='gzip')

        for key, value in extraction_metadata.items():
            f.create_dataset('metadata/extraction/{}'.format(key), data=value)

        video_pipe = None

        for i, frame_range in enumerate(tqdm.tqdm(frame_batches, desc='Processing batches')):
            raw_frames = load_movie_data(args['input_file'], frame_range)
            raw_frames = bground_im-raw_frames
            raw_frames[np.logical_or(raw_frames < args['min_height'], raw_frames > args['max_height'])] = 0
            raw_frames = raw_frames.astype('uint8')
            raw_frames = apply_roi(raw_frames, roi)

            results = extract_chunk(raw_frames,
                                    use_em_tracker=args['em_tracking'],
                                    prefilter_space=args['prefilter_space'],
                                    prefilter_time=args['prefilter_time'],
                                    min_height=args['min_height'],
                                    max_height=args['max_height'],
                                    flip_classifier=args['flip_file'],
                                    crop_size=args['crop_size'])

            # if desired, write out a movie
            # TODO: cut out first part of overhang
            offset = args['chunk_overlap'] if i > 0 else 0

            frame_range = frame_range[offset:]

            for scalar in scalars:
                f['scalars/{}'.format(scalar)][frame_range] = results['scalars'][scalar][offset:, ...]
            f['frames'][frame_range] = results['depth_frames'][offset:, ...]

            nframes, rows, cols = raw_frames[offset:, ...].shape
            output_movie = np.zeros((nframes, rows+args['crop_size'][0], cols+args['crop_size'][1]), 'uint16')
            output_movie[:, :args['crop_size'][0], :args['crop_size'][1]] = results['depth_frames'][offset:, ...]
            output_movie[:, args['crop_size'][0]:, args['crop_size'][1]:] = raw_frames[offset:, ...]

            video_pipe = write_frames_preview(
                os.path.join(output_dir, 'results.mp4'), output_movie,
                pipe=video_pipe, close_pipe=False)

        if video_pipe:
            video_pipe.stdin.close()
            video_pipe.wait()

    print('\n')

@cli.command(name='gen-config')
@click.argument('fpath', type=click.Path(resolve_path=True))
def gen_config(fpath):
    print('Saving default configuration file')
    output = config.save_config(fpath)
    print('Complete: saved to:\n    {}'.format(output))


if __name__ == '__main__':
    cli()
