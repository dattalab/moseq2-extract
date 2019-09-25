from .cli import *
import ruamel.yaml as yaml
from moseq2_extract.io.video import (get_movie_info, load_movie_data,
                                     write_frames_preview, write_frames)
import urllib
import h5py
import tqdm
from moseq2_extract.io.image import write_image, read_image
from moseq2_extract.extract.extract import extract_chunk
from moseq2_extract.extract.proc import apply_roi, get_roi, get_bground_im_file
from moseq2_extract.util import (load_metadata, gen_batch_sequence, load_timestamps,
                                 select_strel, command_with_config, scalar_attributes,
                                 save_dict_contents_to_h5, click_param_annot,
                                 convert_raw_to_avi_function)

def generate_config_command(output_file, gui_options={}):
    if gui_options == {}:
        objs = extract.params
        params = {tmp.name: tmp.default for tmp in objs if not tmp.required}

        with open(output_file, 'w') as f:
            yaml.dump(params, f, Dumper=yaml.RoundTripDumper)
    else:
        params = gui_options
        with open(output_file, 'w') as f:
            yaml.dump(params, f, Dumper=yaml.RoundTripDumper)

def download_flip_command(output_dir, selected_flip):


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

    if selected_flip is not None:
        selection = selected_flip

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

def convert_raw_to_avi_command(input_file, output_file, chunk_size, fps, delete, threads):

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

    return True

def copy_slice_command(input_file, output_file, copy_slice, chunk_size, fps, delete, threads):

    if output_file is None:
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        avi_encode = True
        output_file = os.path.join(os.path.dirname(input_file), '{}.avi'.format(base_filename))
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
        raise RuntimeError('Output file {} already exists'.format(output_file))

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
            raise RuntimeError('Raw frames and encoded frames not equal from {} to {}'.format(batch[0], batch[-1]))

    print('Encoding successful')

    if delete:
        print('Deleting {}'.format(input_file))
        os.remove(input_file)

    return True

def find_roi_command(input_file, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
             bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes,
             bg_sort_roi_by_position, bg_sort_roi_by_position_max_rois,
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
                                  gradient_kernel=bg_roi_gradient_kernel,
                                  fill_holes=bg_roi_fill_holes)

    if bg_sort_roi_by_position:
        rois = rois[:bg_sort_roi_by_position_max_rois]
        rois = [rois[i] for i in np.argsort([np.nonzero(roi)[0].mean() for roi in rois])]

    bg_roi_index = [idx for idx in bg_roi_index if idx in range(len(rois))]
    for idx in bg_roi_index:
        roi_filename = 'roi_{:02d}.tiff'.format(idx)
        write_image(os.path.join(output_dir, roi_filename),
                    rois[idx], scale=True, dtype='uint8')

    return True


def extract_command(input_file, output_dir, config_file):

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)


    print('Processing: {}'.format(input_file))
    # get the basic metadata

    status_dict = {
        'parameters': deepcopy(config_data),
        'complete': False,
        'skip': False,
        'uuid': str(uuid.uuid4()),
        'metadata': ''
    }

    # np.seterr(invalid='raise')

    # handle tarball stuff
    dirname = os.path.dirname(input_file)

    if input_file.endswith('.tar.gz') or input_file.endswith('.tgz'):
        print('Scanning tarball {} (this will take a minute)'.format(input_file))
        #compute NEW psuedo-dirname now, `input_file` gets overwritten below with test_vid.dat tarinfo...
        dirname = os.path.join(dirname, os.path.basename(input_file).replace('.tar.gz', '').replace('.tgz', ''))

        tar = tarfile.open(input_file, 'r:gz')
        tar_members = tar.getmembers()
        tar_names = [_.name for _ in tar_members]
        input_file = tar_members[tar_names.index('test_vid.dat')]
    else:
        tar = None
        tar_members = None

    video_metadata = get_movie_info(input_file)
    nframes = video_metadata['nframes']

    # if tar is not None:
    #     # convert TarInfo into bufferedreader
    #     input_file = tar.extractfile(input_file)

    if config_data['frame_trim'][0] > 0 and config_data['frame_trim'][0] < nframes:
        first_frame_idx = config_data['frame_trim'][0]
    else:
        first_frame_idx = 0

    if nframes - config_data['frame_trim'][1] > first_frame_idx:
        last_frame_idx = nframes - config_data['frame_trim'][1]
    else:
        last_frame_idx = nframes

    nframes = last_frame_idx - first_frame_idx
    alternate_correct = False

    if tar is not None:
        metadata_path = tar.extractfile(tar_members[tar_names.index('metadata.json')])
        if "depth_ts.txt" in tar_names:
            timestamp_path = tar.extractfile(tar_members[tar_names.index('depth_ts.txt')])
        elif "timestamps.csv" in tar_names:
            timestamp_path = tar.extractfile(tar_members[tar_names.index('timestamps.csv')])
            alternate_correct = True
    else:
        metadata_path = os.path.join(dirname, 'metadata.json')
        timestamp_path = os.path.join(dirname, 'depth_ts.txt')
        alternate_timestamp_path = os.path.join(dirname, 'timestamps.csv')
        if not os.path.exists(timestamp_path) and os.path.exists(alternate_timestamp_path):
            timestamp_path = alternate_timestamp_path
            alternate_correct = True

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
        bground_im = get_bground_im_file(input_file, tar_object=tar)
        if not config_data['use_plane_bground']:
            write_image(os.path.join(output_dir, 'bground.tiff'), bground_im, scale=True)

    first_frame = load_movie_data(input_file, 0, tar_object=tar)
    write_image(os.path.join(output_dir, 'first_frame.tiff'), first_frame, scale=True,
                scale_factor=config_data['bg_roi_depth_range'])

    roi_filename = 'roi_{:02d}.tiff'.format(config_data['bg_roi_index'])

    strel_dilate = select_strel((config_data['bg_roi_shape'], config_data['bg_roi_dilate']))
    strel_tail = select_strel((config_data['tail_filter_shape'], config_data['tail_filter_size']))
    strel_min = select_strel((config_data['cable_filter_shape'], config_data['cable_filter_size']))

    if os.path.exists(os.path.join(output_dir, roi_filename)):
        print('Loading ROI...')
        roi = read_image(os.path.join(output_dir, roi_filename), scale=True) > 0
    else:
        print('Getting roi...')
        rois, plane, _, _, _, _ = get_roi(bground_im,
                                          strel_dilate=strel_dilate,
                                          weights=config_data['bg_roi_weights'],
                                          depth_range=config_data['bg_roi_depth_range'],
                                          gradient_filter=config_data['bg_roi_gradient_filter'],
                                          gradient_threshold=config_data['bg_roi_gradient_threshold'],
                                          gradient_kernel=config_data['bg_roi_gradient_kernel'],
                                          fill_holes=config_data['bg_roi_fill_holes'])

        if config_data['use_plane_bground']:
            print('Using plane fit for background...')
            xx, yy = np.meshgrid(np.arange(bground_im.shape[1]), np.arange(bground_im.shape[0]))
            coords = np.vstack((xx.ravel(), yy.ravel()))
            plane_im = (np.dot(coords.T, plane[:2]) + plane[3]) / -plane[2]
            plane_im = plane_im.reshape(bground_im.shape)
            write_image(os.path.join(output_dir, 'bground.tiff'), plane_im, scale=True)
            bground_im = plane_im

        roi = rois[config_data['bg_roi_index']]
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
            f['timestamps'].attrs['description'] = "Depth video timestamps"

        f.create_dataset('frames', (nframes, config_data['crop_size'][0], config_data['crop_size'][1]), config_data['frame_dtype'], compression='gzip')
        f['frames'].attrs['description'] = '3D Numpy array of depth frames (nframes x w x h, in mm)'

        if config_data['use_tracking_model']:
            f.create_dataset('frames_mask', (nframes, config_data['crop_size'][0], config_data['crop_size'][1]), 'float32', compression='gzip')
            f['frames_mask'].attrs['description'] = 'Log-likelihood values from the tracking model (nframes x w x h)'
        else:
            f.create_dataset('frames_mask', (nframes, config_data['crop_size'][0], config_data['crop_size'][1]), 'bool', compression='gzip')
            f['frames_mask'].attrs['description'] = 'Boolean mask, false=not mouse, true=mouse'

        if config_data['flip_classifier'] is not None:
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

        extract_version = np.string_(get_distribution('moseq2-extract').version)
        f.create_dataset('metadata/extraction/extract_version', data=extract_version)
        f['metadata/extraction/extract_version'].attrs['description'] = 'Version of moseq2-extract'

        save_dict_contents_to_h5(f, status_dict['parameters'], 'metadata/extraction/parameters', click_param_annot(extract))

        for key, value in acquisition_metadata.items():
            if type(value) is list and len(value) > 0 and type(value[0]) is str:
                value = [n.encode('utf8') for n in value]

            if value is not None:
                f.create_dataset('metadata/acquisition/{}'.format(key), data=value)
            else:
                f.create_dataset('metadata/acquisition/{}'.format(key), dtype="f")

        video_pipe = None
        tracking_init_mean = None
        tracking_init_cov = None

        for i, frame_range in enumerate(tqdm.tqdm(frame_batches, desc='Processing batches')):
            raw_frames = load_movie_data(input_file, [f + first_frame_idx for f in frame_range], tar_object=tar)
            raw_frames = bground_im-raw_frames
            # raw_frames[np.logical_or(raw_frames < min_height, raw_frames > max_height)] = 0
            raw_frames[raw_frames < config_data['min_height']] = 0
            raw_frames[raw_frames > config_data['max_height']] = config_data['max_height']
            raw_frames = raw_frames.astype(config_data['frame_dtype'])
            raw_frames = apply_roi(raw_frames, roi)

            results = extract_chunk(raw_frames,
                                    use_em_tracker=config_data['use_tracking_model'],
                                    strel_tail=strel_tail,
                                    strel_min=strel_min,
                                    iters_tail=config_data['tail_filter_iters'],
                                    iters_min=config_data['cable_filter_iters'],
                                    prefilter_space=config_data['spatial_filter_size'],
                                    prefilter_time=config_data['temporal_filter_size'],
                                    min_height=config_data['min_height'],
                                    max_height=config_data['max_height'],
                                    flip_classifier=config_data['flip_classifier'],
                                    flip_smoothing=config_data['flip_classifier_smoothing'],
                                    crop_size=config_data['crop_size'],
                                    frame_dtype=config_data['frame_dtype'],
                                    mask_threshold=config_data['tracking_model_mask_threshold'],
                                    tracking_ll_threshold=config_data['tracking_model_ll_threshold'],
                                    tracking_segment=config_data['tracking_model_segment'],
                                    tracking_init_mean=tracking_init_mean,
                                    tracking_init_cov=tracking_init_cov,
                                    true_depth=true_depth,
                                    centroid_hampel_span=config_data['centroid_hampel_span'],
                                    centroid_hampel_sig=config_data['centroid_hampel_sig'],
                                    angle_hampel_span=config_data['angle_hampel_span'],
                                    angle_hampel_sig=config_data['angle_hampel_sig'],
                                    model_smoothing_clips=config_data['model_smoothing_clips'],
                                    tracking_model_init=config_data['tracking_model_init'])

            # if desired, write out a movie

            if i > 0:
                offset = config_data['chunk_overlap']
            else:
                offset = 0

            if config_data['use_tracking_model']:
                results['mask_frames'][results['depth_frames'] < config_data['min_height']] = config_data['tracking_model_ll_clip']
                results['mask_frames'][results['mask_frames'] < config_data['tracking_model_ll_clip']] = config_data['tracking_model_ll_clip']
                tracking_init_mean = results['parameters']['mean'][-(config_data['chunk_overlap']+1)]
                tracking_init_cov = results['parameters']['cov'][-(config_data['chunk_overlap']+1)]

            frame_range = frame_range[offset:]

            for scalar in scalars:
                f['scalars/{}'.format(scalar)][frame_range] = results['scalars'][scalar][offset:, ...]

            f['frames'][frame_range] = results['depth_frames'][offset:, ...]
            f['frames_mask'][frame_range] = results['mask_frames'][offset:, ...]

            if config_data['flip_classifier']:
                f['metadata/extraction/flips'][frame_range] = results['flips'][offset:]

            nframes, rows, cols = raw_frames[offset:, ...].shape
            output_movie = np.zeros((nframes, rows+config_data['crop_size'][0], cols+config_data['crop_size'][1]), 'uint16')
            output_movie[:, :config_data['crop_size'][0], :config_data['crop_size'][1]] = results['depth_frames'][offset:, ...]
            output_movie[:, config_data['crop_size'][0]:, config_data['crop_size'][1]:] = raw_frames[offset:, ...]

            video_pipe = write_frames_preview(
                os.path.join(output_dir, '{}.mp4'.format(output_filename)), output_movie,
                pipe=video_pipe, close_pipe=False, fps=config_data['fps'],
                frame_range=[f + first_frame_idx for f in frame_range],
                depth_max=config_data['max_height'], depth_min=config_data['min_height'])

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
        yaml.dump(status_dict, f, Dumper=yaml.RoundTripDumper)
