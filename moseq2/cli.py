from moseq2.io.video import get_movie_info,\
    load_movie_data, write_frames_preview
from moseq2.io.image import write_image, read_image
from moseq2.extract.extract import extract_chunk
from moseq2.extract.proc import apply_roi, get_roi, get_bground_im_file
from moseq2.util import load_metadata, gen_batch_sequence, load_timestamps,\
    select_strel, build_path, h5_to_dict, recursive_find_h5s, camel_to_snake,\
    recursive_find_unextracted_dirs
from copy import deepcopy
import click
import os
import h5py
import tqdm
import numpy as np
import ruamel.yaml as yaml
import uuid
import pathlib
import datetime


# from https://stackoverflow.com/questions/46358797/
# python-click-supply-arguments-and-options-from-a-configuration-file
def CommandWithConfigFile(config_file_param_name):

    class CustomCommandClass(click.Command):

        def invoke(self, ctx):
            config_file = ctx.params[config_file_param_name]
            if config_file is not None:
                with open(config_file) as f:
                    config_data = yaml.load(f, yaml.Loader)
                    for param, value in ctx.params.items():
                        if param in config_data:
                            ctx.params[param] = config_data[param]

            return super(CustomCommandClass, self).invoke(ctx)

    return CustomCommandClass


@click.group()
def cli():
    pass


@cli.command(name="find-roi", cls=CommandWithConfigFile('config_file'))
@click.argument('input-file', type=click.Path(exists=True))
@click.option('--roi-dilate', default=(10, 10), type=(int, int), help='Size of strel to dilate roi')
@click.option('--roi-shape', default='ellipse', type=str, help='Shape to use to dilate roi (ellipse or rect)')
@click.option('--roi-index', default=0, type=int, help='Index of roi to use', multiple=True)
@click.option('--roi-weights', default=(1, .1, 1), type=(float, float, float),
              help='ROI feature weighting (area, extent, dist)')
@click.option('--output-dir', default=None, help='Output directory')
@click.option('--use-plane-bground', default=False, type=bool, help='Use plane fit for background')
@click.option("--config-file", type=click.Path())
def find_roi(input_file, roi_dilate, roi_shape, roi_index, roi_weights,
             output_dir, use_plane_bground, config_file):

    # set up the output directory

    if type(roi_index) is int:
        roi_index = [roi_index]

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

    rois, _, _, _, _, _ = get_roi(bground_im, strel_dilate=strel_dilate,
                                  weights=roi_weights)
    for idx in roi_index:
        roi_filename = 'roi_{:02d}.tiff'.format(idx)
        write_image(os.path.join(output_dir, roi_filename),
                    rois[idx], scale=True, dtype='uint8')


@cli.command(name="extract", cls=CommandWithConfigFile('config_file'))
@click.argument('input-file', type=click.Path(exists=True))
@click.option('--crop-size', '-c', default=(80, 80), type=(int, int),
              help='Width and height of cropped mouse')
@click.option('--roi-dilate', default=(10, 10), type=(int, int), help='Size of strel to dilate roi')
@click.option('--roi-shape', default='ellipse', type=str, help='Shape to use to dilate roi (ellipse or rect)')
@click.option('--roi-index', default=0, type=int, help='Index of roi to use', multiple=True)
@click.option('--roi-weights', default=(1, .1, 1), type=(float, float, float),
              help='ROI feature weighting (area, extent, dist)')
@click.option('--min-height', default=10, type=int, help='Min height of mouse from floor (mm)')
@click.option('--max-height', default=100, type=int, help='Max height of mouse from floor (mm)')
@click.option('--fps', default=30, type=int, help='Frame rate of camera')
@click.option('--flip-file', default=None, help='Location of flip classifier (.pkl)')
@click.option('--em-tracking', is_flag=True, help='Use em tracker')
@click.option('--minfilter-iters', default=0, type=int, help="Number of minimum filter iterations")
@click.option('--minfilter-shape', default='rectangle', type=str, help="Minimum filter shape")
@click.option('--minfilter-size', default=(5, 5), type=(int, int), help="Minimum filter size")
@click.option('--tailfilter-iters', default=1, type=int, help="Number of tail filter iterations")
@click.option('--tailfilter-size', default=(9, 9), type=(int, int), help='Tail filter size')
@click.option('--tailfilter-shape', default='ellipse', type=str, help='Tail filter shape')
@click.option('--prefilter-space', default=(3,), type=tuple, help='Space prefilter kernel')
@click.option('--prefilter-time', default=(), type=tuple, help='Time prefilter kernel')
@click.option('--chunk-size', default=1000, type=int, help='Chunk size for processing')
@click.option('--chunk-overlap', default=60, type=int, help='Overlap in chunks')
@click.option('--output-dir', default=None, help='Output directory')
@click.option('--write-movie', default=True, type=bool, help='Write results movie')
@click.option('--use-plane-bground', is_flag=True, help='Use plane fit for background')
@click.option("--config-file", type=click.Path())
def extract(input_file, crop_size, roi_dilate, roi_shape, roi_weights, roi_index,
            min_height, max_height, fps, flip_file, em_tracking, minfilter_iters,
            minfilter_shape, minfilter_size, tailfilter_iters, tailfilter_size,
            tailfilter_shape, prefilter_space, prefilter_time, chunk_size, chunk_overlap,
            output_dir, write_movie, use_plane_bground, config_file):

    # get the basic metadata

    # if we pass in multiple roi indices, recurse and process each roi
    # if len(roi_index) > 1:
    #     for roi in roi_index:
    #         extract(roi_index=roi, **locals())
    #     return None

    status_dict = {
        'parameters': locals(),
        'complete': False,
        'skip': False,
        'uuid': str(uuid.uuid4())
    }

    np.seterr(invalid='raise')

    video_metadata = get_movie_info(input_file)
    nframes = video_metadata['nframes']
    extraction_metadata = load_metadata(os.path.join(os.path.dirname(input_file), 'metadata.json'))
    timestamps = load_timestamps(os.path.join(os.path.dirname(input_file), 'depth_ts.txt'), col=0)

    scalars = ['centroid_x', 'centroid_y', 'angle', 'width',
               'length', 'height_ave', 'velocity_mag',
               'velocity_theta', 'area', 'velocity_mag_3d']

    frame_batches = list(gen_batch_sequence(nframes, chunk_size, chunk_overlap))

    # set up the output directory

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_file), 'proc')
    else:
        output_dir = os.path.join(os.path.dirname(input_file), output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = 'results_{:02d}'.format(roi_index)
    status_filename = os.path.join(output_dir, '{}.yaml'.format(output_filename))

    if os.path.exists(status_filename):
        raise RuntimeError("Already found a status file in {}, delete and try again".format(status_filename))

    with open(status_filename, 'w') as f:
        yaml.dump(status_dict, f)

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
                scale_factor=(650, 750))

    roi_filename = 'roi_{:02d}.tiff'.format(roi_index)

    strel_dilate = select_strel(roi_shape, roi_dilate)
    strel_tail = select_strel(tailfilter_shape, tailfilter_size)
    strel_min = select_strel(minfilter_shape, minfilter_size)

    if os.path.exists(os.path.join(output_dir, roi_filename)):
        print('Loading ROI...')
        roi = read_image(os.path.join(output_dir, roi_filename), scale=True) > 0
    else:
        print('Getting roi...')
        rois, plane, _, _, _, _ = get_roi(bground_im, strel_dilate=strel_dilate,
                                          weights=roi_weights)

        if use_plane_bground:
            print('Using plane fit for background...')
            xx, yy = np.meshgrid(np.arange(bground_im.shape[1]), np.arange(bground_im.shape[0]))
            coords = np.vstack((xx.ravel(), yy.ravel()))
            plane_im = (np.dot(coords.T, plane[:2]) + plane[3]) / -plane[2]
            plane_im = plane_im.reshape(bground_im.shape)
            write_image(os.path.join(output_dir, 'bground.tiff'), plane_im, scale=True)
            bground_im = plane_im

        roi = rois[roi_index]
        write_image(os.path.join(output_dir, roi_filename),
                    roi, scale=True, dtype='uint8')

    # farm out the batches and write to an hdf5 file

    with h5py.File(os.path.join(output_dir, '{}.h5'.format(output_filename)), 'w') as f:
        for i in range(len(scalars)):
            f.create_dataset('scalars/{}'.format(scalars[i]), (nframes,), 'float32', compression='gzip')

        f.create_dataset('metadata/timestamps', compression='gzip', data=timestamps)
        f.create_dataset('frames', (nframes, crop_size[0], crop_size[1]), 'i1', compression='gzip')

        for key, value in extraction_metadata.items():
            f.create_dataset('metadata/extraction/{}'.format(key), data=value)

        video_pipe = None

        for i, frame_range in enumerate(tqdm.tqdm(frame_batches, desc='Processing batches')):
            raw_frames = load_movie_data(input_file, frame_range)
            raw_frames = bground_im-raw_frames
            raw_frames[np.logical_or(raw_frames < min_height, raw_frames > max_height)] = 0
            raw_frames = raw_frames.astype('uint8')
            raw_frames = apply_roi(raw_frames, roi)

            results = extract_chunk(raw_frames,
                                    use_em_tracker=em_tracking,
                                    strel_tail=strel_tail,
                                    strel_min=strel_min,
                                    iters_tail=tailfilter_iters,
                                    iters_min=minfilter_iters,
                                    prefilter_space=prefilter_space,
                                    prefilter_time=prefilter_time,
                                    min_height=min_height,
                                    max_height=max_height,
                                    flip_classifier=flip_file,
                                    crop_size=crop_size)

            # if desired, write out a movie

            # todo: cut out first part of overhang

            if i > 0:
                offset = chunk_overlap
            else:
                offset = 0

            frame_range = frame_range[offset:]

            for scalar in scalars:
                f['scalars/{}'.format(scalar)][frame_range] = results['scalars'][scalar][offset:, ...]
            f['frames'][frame_range] = results['depth_frames'][offset:, ...]

            nframes, rows, cols = raw_frames[offset:, ...].shape
            output_movie = np.zeros((nframes, rows+crop_size[0], cols+crop_size[1]), 'uint16')
            output_movie[:, :crop_size[0], :crop_size[1]] = results['depth_frames'][offset:, ...]
            output_movie[:, crop_size[0]:, crop_size[1]:] = raw_frames[offset:, ...]

            video_pipe = write_frames_preview(
                os.path.join(output_dir, '{}.mp4'.format(output_filename)), output_movie,
                pipe=video_pipe, close_pipe=False)

        if video_pipe:
            video_pipe.stdin.close()
            video_pipe.wait()

    status_dict['complete'] = True

    with open(status_filename, 'w') as f:
        yaml.dump(status_dict, f)

    print('\n')


# recurse through directories, find h5 files with completed extractions, make a manifest
# and copy the contents to a new directory
@cli.command(name="aggregate-results")
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--format', '-f', type=str, default='${start_time}_${session_name}_${subject_name}',
              help="Mapping from metadata to new file")
@click.option('--output-dir', '-o', type=click.Path(),
              default=os.path.join(os.getcwd(), 'aggregate_results/'),
              help="Where to store results")
def aggregate_results(input_dir, format, output_dir):

    h5s, dicts, yamls = recursive_find_h5s(input_dir)
    to_load = [(tmp, file) for tmp, file in zip(dicts, h5s) if tmp['complete'] and not tmp['skip']]

    # load in all of the h5 files, grab the extraction metadata, reformat to make nice 'n pretty
    # then stage the copy

    for i, tup in enumerate(to_load):
        with h5py.File(tup[1], 'r') as f:
            tmp = h5_to_dict(f, '/metadata/extraction')
            tmp2 = deepcopy(tmp)
            for key, val in tmp.items():
                tmp2[camel_to_snake(key)] = tmp2.pop(key)
            to_load[i][0]['extraction_metadata'] = tmp2

    manifest = {}

    for tup in to_load:
        copy_path = build_path(tup[0]['extraction_metadata'], format)
        manifest[tup[1]] = copy_path

    print(manifest)


@cli.command(name="extract-batch")
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--config-file', '-c', type=click.Path(), help="Path to configuration file")
@click.option('--cluster-type', type=click.Choice(['slurm']), default='slurm', help='Cluster type')
@click.option('--temp-storage', type=click.Path(),
              default=os.path.join(pathlib.Path.home(), 'moseq2'), help="Temp storage")
@click.option('--ncpus', type=int, default=1, help="Number of CPUs")
@click.option('--mem', type=int, default=5000, help="RAM in MB")
@click.option('--wall-time', type=str, default='3:00:00', help="Wall time")
@click.option('--partition', type=str, default='short', help="Partition name")
@click.option('--prefix', type=str, default='source activate moseq2', help="Command to run before extract")
def extract_batch(input_dir, config_file, cluster_type, temp_storage,
                  ncpus, mem, wall_time, partition, prefix):
    # find directories with .dat files that either have incomplete or no extractions

    to_extract = recursive_find_unextracted_dirs(input_dir)
    objs = extract.params
    params = {tmp.name: tmp.default for tmp in objs if not tmp.required}
    param_names = list(params.keys())

    if config_file is not None and not os.path.exists(config_file):
        raise IOError('Config file {} does not exist'.format(config_file))
    elif config_file is not None:
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            for k, v in config.items():
                if k in param_names:
                    params[k] = v

    if not os.path.exists(temp_storage):
        os.makedirs(temp_storage)

    suffix = '_{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())

    config_store = os.path.join(temp_storage, 'job_config{}.yaml'.format(suffix))
    with open(config_store, 'w') as f:
        yaml.dump(params, f)

    if cluster_type == 'slurm':

        base_command = 'sbatch -n={:d} --mem={:d}M --partition={} -t={} --wrap "'\
            .format(ncpus, mem, partition, wall_time)
        if prefix is not None:
            base_command += '{}; '.format(prefix)

        for ext in to_extract:

            if len(config['roi_index']) > 1:
                base_command += 'moseq2 find-roi --config-file {} {}; '.format(config_store, ext)

            for roi in config['roi_index']:
                roi_config = deepcopy(params)
                roi_config['roi_index'] = roi
                roi_config_store = os.path.join(temp_storage, 'job_config{}_roi{:d}.yaml'.format(suffix, roi))
                with open(roi_config_store, 'w') as f:
                    yaml.dump(roi_config, f)

                base_command += 'moseq2 extract --config-file {} --roi-index {:d} {}; '\
                    .format(roi_config_store, roi, ext)

            base_command += '"'
            print(base_command)

    else:
        raise NotImplementedError('Other cluster types not supported')


if __name__ == '__main__':
    cli()
