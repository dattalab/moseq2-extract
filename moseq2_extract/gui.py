from .cli import *
import ruamel.yaml as yaml
from moseq2_extract.io.video import (get_movie_info, load_movie_data,
                                     write_frames, convert_mkv_to_avi)
import urllib
import h5py
import tqdm
import re
import os
import warnings
from pathlib import Path
from glob import glob
import json
from cytoolz import partial
from moseq2_extract.command_helpers.data import get_selected_sessions, load_h5s, build_manifest, copy_manifest_results,\
                                                handle_extract_metadata, create_extract_h5
from moseq2_extract.command_helpers.extract import run_local_extract, run_slurm_extract, process_extract_batches,\
                                                    extract_roi_helper, get_roi_helper
from moseq2_extract.io.image import write_image, read_image

from moseq2_extract.util import (load_metadata, gen_batch_sequence, load_timestamps,
                                 select_strel, scalar_attributes,
                                 convert_raw_to_avi_function, recursive_find_h5s, escape_path,
                                 mouse_threshold_filter, recursive_find_unextracted_dirs)
from moseq2_pca.cli import train_pca, apply_pca, compute_changepoints
from moseq2_model.cli import learn_model, count_frames
from moseq2_viz.cli import make_crowd_movies, plot_transition_graph


def update_progress(progress_file, varK, varV):
    with open(progress_file, 'r') as f:
        progress = yaml.safe_load(f)
    f.close()

    progress[varK] = varV
    with open(progress_file, 'w') as f:
        yaml.safe_dump(progress, f)

    print(f'Successfully updated progress file with {varK} -> {varV}')

def restore_progress_vars(progress_file):
    with open(progress_file, 'r') as f:
        vars = yaml.safe_load(f)
    f.close()

    return vars['config_file'], vars['index_file'], vars['train_data_dir'], vars['pca_dirname'], vars['scores_filename'], vars['model_path'], vars['scores_path'], vars['crowd_dir'], vars['plot_path']

def check_progress(base_dir, progress_filepath, output_directory=None):

    if output_directory is not None:
        progress_filepath = os.path.join(output_directory, progress_filepath.split('/')[-1])

    if os.path.exists(progress_filepath):
        with open(progress_filepath, 'r') as f:
            progress_vars = yaml.safe_load(f)
        f.close()

        print('Progress file found, listing initialized variables...\n')

        for k, v in progress_vars.items():
            if v != 'TBD':
                print(f'{k}: {v}')

        restore = ''
        while(restore != 'Y' or restore != 'N' or restore != 'q'):
            restore = input('Would you like to restore the above listed notebook variables? [Y -> restore variables, N -> overwrite progress file, q -> quit]')
            if restore == "Y":

                print('Updating notebook variables...')

                config_filepath, index_filepath, train_data_dir, pca_dirname, \
                scores_filename, model_path, scores_file, \
                crowd_dir, plot_path = restore_progress_vars(progress_filepath)

                return config_filepath, index_filepath, train_data_dir, pca_dirname, \
                scores_filename, model_path, scores_file, \
                crowd_dir, plot_path
            elif restore == "N":

                print('Overwriting progress file.')

                progress_vars = {'base_dir': base_dir, 'config_file': 'TBD', 'index_file': 'TBD', 'train_data_dir': 'TBD',
                                 'pca_dirname': 'TBD', 'scores_filename': 'TBD', 'scores_path': 'TBD', 'model_path': 'TBD',
                                 'crowd_dir': 'TBD', 'plot_path': 'TBD'}

                with open(progress_filepath, 'w') as f:
                    yaml.safe_dump(progress_vars, f)
                f.close()

                print('\nProgress file created, listing initialized variables...')
                for k, v in progress_vars.items():
                    if v != 'TBD':
                        print(k, v)
                return progress_vars['config_file'], progress_vars['index_file'], progress_vars['train_data_dir'], progress_vars['pca_dirname'], progress_vars['scores_filename'], \
                       progress_vars['model_path'], progress_vars['scores_path'], progress_vars['crowd_dir'], progress_vars['plot_path']
            elif restore == 'q':
                return


    else:
        print('Progress file not found, creating new one.')
        progress_vars = {'base_dir': base_dir, 'config_file': 'TBD', 'index_file': 'TBD', 'train_data_dir': 'TBD', 'pca_dirname': 'TBD',
                         'scores_filename': 'TBD', 'scores_path': 'TBD', 'model_path': 'TBD', 'crowd_dir': 'TBD', 'plot_path': 'TBD'}

        with open(progress_filepath, 'w') as f:
            yaml.safe_dump(progress_vars, f)
        f.close()

        print('\nProgress file created, listing initialized variables...')
        for k, v in progress_vars.items():
            if v != 'TBD':
                print(k, v)

        return progress_vars['config_file'], progress_vars['index_file'], progress_vars['train_data_dir'], progress_vars['pca_dirname'], progress_vars['scores_filename'],\
               progress_vars['model_path'], progress_vars['scores_path'], progress_vars['crowd_dir'], progress_vars['plot_path']

def generate_config_command(output_file):
    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    objs = extract.params
    objs2, objs3, objs4, objs5 = find_roi.params, train_pca.params, apply_pca.params, compute_changepoints.params
    objsM, objsF = learn_model.params, count_frames.params
    objsV1, objsV2 = make_crowd_movies.params, plot_transition_graph.params

    objsT = objs2+objs3+objs4+objs5+objsM+objsF+objsV1+objsV2

    params = {tmp.name: tmp.default for tmp in objs if not tmp.required}
    for obj in objsT:
        if obj.name not in params.keys():
            params[obj.name] = obj.default

    input_dir = os.path.dirname(output_file)
    params['input_dir'] = input_dir
    params['detected_true_depth'] = 'auto'
    if os.path.exists(output_file):
        print('This file already exists, would you like to overwrite it? [Y -> yes, else -> exit]')
        ow = input()
        if ow == 'Y':
            with open(output_file, 'w') as f:
                yaml.safe_dump(params, f)
        else:
            return 'Configuration file has been retained'
    else:
        print('Creating configuration file.')
        with open(output_file, 'w') as f:
            yaml.safe_dump(params, f)

    return 'Configuration file has been successfully generated.'

def view_extraction(extractions):

    if len(extractions) > 1:
        for i, sess in enumerate(extractions):
            print(f'[{str(i + 1)}] {sess}')
    else:
        print('no sessions to view')

    while (True):
        try:
            input_file_indices = input(
                "Input extracted session indices to view separated by commas, or empty string for all sessions.\n").strip()
            if ',' in input_file_indices:
                input_file_indices = input_file_indices.split(',')
                for i in input_file_indices:
                    i = int(i.strip())
                    if i > len(extractions):
                        print('invalid index try again.')
                        input_file_index = []
                        break

                tmp = []
                for index in input_file_indices:
                    index = int(index.strip())
                    tmp.append(extractions[index - 1])
                extractions = tmp
                break
            elif len(input_file_indices.strip()) == 1:
                index = int(input_file_indices.strip())
                extractions = [extractions[index - 1]]
                break
            elif input_file_indices == '':
                break
        except:
            print('unexpected error:', sys.exc_info()[0])

    return extractions

def extract_found_sessions(input_dir, config_file, filename, extract_all=True, skip_extracted=False, output_directory=None):
    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # find directories with .dat files that either have incomplete or no extractions
    partition = 'short'
    skip_checks = True
    config_file = Path(config_file)

    prefix = ''
    to_extract = recursive_find_unextracted_dirs(input_dir, filename=filename, skip_checks=skip_checks)
    temp = []
    for dir in to_extract:
        if '/tmp/' not in dir:
            temp.append(dir)

    to_extract = temp

    to_extract = get_selected_sessions(to_extract, extract_all)

    if config_file is None:
        raise RuntimeError('Need a config file to continue')
    elif not os.path.exists(config_file):
        raise IOError(f'Config file {config_file} does not exist')

    with config_file.open() as f:
        params = yaml.safe_load(f)

    cluster_type = params['cluster_type']
    # bg_roi_index = params['bg_roi_index'] ## replaced input parameter with config parameter

    if type(params['bg_roi_index']) is int:
        params['bg_roi_index'] = [params['bg_roi_index']]


    if cluster_type == 'slurm':
        run_slurm_extract(to_extract, params, partition, prefix, escape_path, skip_extracted, output_directory)

    elif cluster_type == 'local':

        run_local_extract(to_extract, params, prefix, skip_extracted, output_directory)

    else:
        raise NotImplementedError('Other cluster types not supported')

    print('Extractions Complete.')

def generate_index_command(input_dir, pca_file, output_file, filter, all_uuids):
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
        with h5py.File(pca_file, 'r') as f:
            pca_uuids = list(f['scores'].keys())

    try:
        file_with_uuids = [(os.path.abspath(h5), os.path.abspath(yml), meta) for h5, yml, meta in
                           zip(h5s, yamls, dicts) if meta['uuid'] in pca_uuids]
    except:
        file_with_uuids = [(os.path.abspath(h5), os.path.abspath(yml), meta) for h5, yml, meta in
                           zip(h5s, yamls, dicts)]
    try:
        if 'metadata' not in file_with_uuids[0][2]:
            raise RuntimeError('Metadata not present in yaml files, run copy-h5-metadata-to-yaml to update yaml files')
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

    print('Index file successfully generated.')
    return output_file


def aggregate_extract_results_command(input_dir, format, output_dir, output_directory=None):
    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    mouse_threshold = 0
    snake_case = True
    if output_directory is None:
        output_dir = os.path.join(input_dir, output_dir)
    else:
        output_dir = os.path.join(output_directory, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    h5s, dicts, _ = recursive_find_h5s(input_dir)

    not_in_output = lambda f: not os.path.exists(os.path.join(output_dir, os.path.basename(f)))
    complete = lambda d: d['complete'] and not d['skip']

    # only include real extracted mice with this filter func
    mtf = partial(mouse_threshold_filter, thresh=mouse_threshold)

    def filter_h5(args):
        '''remove h5's that should be skipped or extraction wasn't complete'''
        _dict, _h5 = args
        return complete(_dict) and not_in_output(_h5) and mtf(_h5)

    # load in all of the h5 files, grab the extraction metadata, reformat to make nice 'n pretty
    # then stage the copy
    to_load = list(filter(filter_h5, zip(dicts, h5s)))

    loaded = load_h5s(to_load)

    manifest = build_manifest(loaded)

    copy_manifest_results(manifest, output_dir)

    print('Results successfully aggregated in', output_dir)

    if output_directory is None:
        indexpath = generate_index_command(output_dir, '', os.path.join(input_dir, 'moseq2-index.yaml'), (), False)
    else:
        indexpath = generate_index_command(input_dir, '', os.path.join(output_directory, 'moseq2-index.yaml'), (), False)

    print(f'Index file path: {indexpath}')
    return indexpath

def get_found_sessions(data_dir="", exts=['dat', 'mkv', 'avi']):
    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    found_sessions = 0
    sessions = []
    for ext in exts:
        if len(data_dir) == 0:
            data_dir = os.getcwd()
            files = sorted(glob('*/*.'+ext))
            found_sessions += len(files)
            sessions += files
        else:
            data_dir = data_dir.strip()
            if os.path.isdir(data_dir):
                files = sorted(glob(os.path.join(data_dir,'*/*.'+ext)))
                found_sessions += len(files)
                sessions += files
            else:
                print('directory not found, try again.')

    # generate sample metadata json for each session that is missing one
    sample_meta = {'SubjectName': 'default', 'SessionName': 'default',
                   'NidaqChannels': 0, 'NidaqSamplingRate': 0.0, 'DepthResolution': [512, 424],
                   'ColorDataType': "Byte[]", "StartTime": ""}
    for sess in sessions:
        sess_dir = '/'.join(sess.split('/')[:-1])
        sess_name = sess.split('/')[-2]
        if 'metadata.json' not in os.listdir(sess_dir):
            sample_meta['SessionName'] = sess_name
            with open(os.path.join(sess_dir, 'metadata.json'), 'w') as fp:
                json.dump(sample_meta, fp)

    for i, sess in enumerate(sessions):
        print(f'[{str(i+1)}] {sess}')

    return data_dir, found_sessions


def download_flip_command(output_dir, config_file="", selection=None):
    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    selected_flip = 1


    flip_files = {
        'large mice with fibers':
            "https://storage.googleapis.com/flip-classifiers/flip_classifier_k2_largemicewithfiber.pkl",
        'adult male c57s':
            "https://storage.googleapis.com/flip-classifiers/flip_classifier_k2_c57_10to13weeks.pkl",
        'mice with Inscopix cables':
            "https://storage.googleapis.com/flip-classifiers/flip_classifier_k2_inscopix.pkl"
    }

    key_list = list(flip_files.keys())

    if selected_flip is not None:
        selection = selected_flip

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    selection = flip_files[key_list[selection]]

    output_filename = os.path.join(output_dir, os.path.basename(selection))
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

    return 'Successfully updated configuration file with adult c57 mouse flip classifier.'


def convert_raw_to_avi_command(input_file, output_file, chunk_size, fps, delete, threads):

    if output_file is None:
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(os.path.dirname(input_file),
                                   '{}.avi'.format(base_filename))

    vid_info = get_movie_info(input_file)
    frame_batches = list(gen_batch_sequence(vid_info['nframes'], chunk_size, 0))
    video_pipe = None

    for batch in tqdm.tqdm_notebook(frame_batches, desc='Encoding batches'):
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

    for batch in tqdm.tqdm_notebook(frame_batches, desc='Checking data integrity'):
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

    for batch in tqdm.tqdm_notebook(frame_batches, desc='Encoding batches'):
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

    for batch in tqdm.tqdm_notebook(frame_batches, desc='Checking data integrity'):
        raw_frames = load_movie_data(input_file, batch)
        encoded_frames = load_movie_data(output_file, batch)

        if not np.array_equal(raw_frames, encoded_frames):
            raise RuntimeError('Raw frames and encoded frames not equal from {} to {}'.format(batch[0], batch[-1]))

    print('Encoding successful')

    if delete:
        print('Deleting {}'.format(input_file))
        os.remove(input_file)

    return True

def find_roi_command(input_dir, config_file, exts=['dat', 'mkv', 'avi'], output_directory=None):
    # set up the output directory
    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    files = []
    for ext in exts:
        files += sorted(glob(os.path.join(input_dir, '*/*.'+ext)))

    for i, sess in enumerate(files):
        print(f'[{str(i+1)}] {sess}')

    input_file_index = -1
    while(int(input_file_index) < 0):
        try:
            input_file_index = int(input("Input session index to find rois: ").strip())
            if int(input_file_index) > len(files):
                print('invalid index try again.')
                input_file_index = -1
        except:
            print('invalid input, only input integers.')

    if output_directory is None:
        output_dir = os.path.join(os.path.dirname(files[input_file_index-1]), 'proc')
    else:
        output_dir = os.path.join(output_directory, 'proc')

    input_file = files[input_file_index-1]

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    if type(config_data['bg_roi_index']) is int:
        bg_roi_index = [config_data['bg_roi_index']]

    if not output_dir:
        output_dir = os.path.join(os.path.dirname(input_file), 'proc')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if input_file.endswith('.mkv'):
        # create a depth.avi file to represent PCs
        temp = convert_mkv_to_avi(input_file)
        if isinstance(temp, str):
            input_file = temp

    get_roi_helper(input_file, bg_roi_index, config_data, output_dir)

    images = []
    filenames = []
    for infile in os.listdir(output_dir):
        if infile[-4:] == "tiff":
            im = read_image(os.path.join(output_dir, infile))
            if len(im.shape) == 2:
                images.append(im)
            elif len(im.shape) == 3:
                images.append(im[0])
            filenames.append(infile)

    print(f'ROIs were successfully computed in {output_dir}')
    return images, filenames


def sample_extract_command(input_dir, config_file, nframes, output_directory=None, exts=['dat', 'mkv', 'avi']):
    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    files = []
    for ext in exts:
        files += sorted(glob(os.path.join(input_dir, '*/*.'+ext)))
    files = sorted(files)
    for i, sess in enumerate(files):
        print(f'[{str(i + 1)}] {sess}')
    input_file_index = -1
    while (int(input_file_index) < 0):
        try:
            input_file_index = int(input("Input session index to extract sample: ").strip())
            if int(input_file_index) > len(files):
                print('invalid index try again.')
                input_file_index = -1
        except:
            print('invalid input, only input integers.')

    if output_directory is None:
        output_dir = os.path.join(os.path.dirname(files[input_file_index-1]), 'sample_proc')
    else:
        output_dir = os.path.join(output_directory, 'sample_proc')

    input_file = files[input_file_index - 1]

    extract_command(input_file, output_dir, config_file, num_frames=nframes)
    print(f'Sample extraction of {str(nframes)} frames completed successfully in {output_dir}.')
    return output_dir

def extract_command(input_file, output_dir, config_file, num_frames=None, skip=False):
    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    if config_data['spatial_filter_size'][0] % 2 == 0 and config_data['spatial_filter_size'][0] > 0:
        config_data['spatial_filter_size'][0] += 1
    if config_data['temporal_filter_size'][0] % 2 == 0 and config_data['temporal_filter_size'][0] > 0:
        config_data['temporal_filter_size'][0] += 1

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

    video_metadata = get_movie_info(input_file)
    if num_frames == None:
        nframes = video_metadata['nframes']
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

    roi, bground_im, first_frame = extract_roi_helper(input_file, output_dir, config_data, bg_roi_file, tar)

    if config_data['detected_true_depth'] == 'auto':
        true_depth = np.median(bground_im[roi > 0])
    else:
        true_depth = config_data['detected_true_depth']
    if input_file.endswith('mkv'):
        new_bg = np.ma.masked_not_equal(roi, 0)
        bground_im = np.where(new_bg == True, new_bg, true_depth)

    print('Detected true depth: {}'.format(true_depth))

    # farm out the batches and write to an hdf5 file

    with h5py.File(os.path.join(output_dir, '{}.h5'.format(output_filename)), 'w') as f:

        create_extract_h5(f, acquisition_metadata, config_data, status_dict, scalars, scalars_attrs, nframes,
                          true_depth, roi, bground_im, first_frame, timestamps)

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

    return 'Extraction completed.'