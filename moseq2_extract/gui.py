from .cli import *
import ruamel.yaml as yaml
from moseq2_extract.io.video import (get_movie_info, load_movie_data,
                                     write_frames_preview, write_frames,
                                     convert_mkv_to_avi)
import urllib
import h5py
import tqdm
import os
import warnings
import shutil
import datetime
from pathlib import Path
from PIL import Image
from glob import glob
import json
from cytoolz import keymap, partial, valmap, assoc
from moseq2_extract.io.image import write_image, read_image
from moseq2_extract.extract.extract import extract_chunk
from moseq2_extract.extract.proc import apply_roi, get_roi, get_bground_im_file
from moseq2_extract.util import (load_metadata, gen_batch_sequence, load_timestamps,
                                 select_strel, command_with_config, scalar_attributes,
                                 save_dict_contents_to_h5, click_param_annot,
                                 convert_raw_to_avi_function, recursive_find_h5s, escape_path,
                                 clean_file_str, load_textdata, time_str_for_filename, build_path,
                                 read_yaml, mouse_threshold_filter, _load_h5_to_dict, h5_to_dict, camel_to_snake,
                                 recursive_find_unextracted_dirs, graduate_dilated_wall_area)
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

    if len(to_extract) > 1 and not extract_all:
        for i, sess in enumerate(to_extract):
            print(f'[{str(i + 1)}] {sess}')

        input_file_indices = []
        while (True):
            try:
                input_file_indices = input("Input session indices to extract separated by commas, or empty string for all sessions.\nTo extract all except certain sessions: type 'e index1,index2,...'\n").strip()
                if 'e' not in input_file_indices:
                    if ',' in input_file_indices:
                        input_file_indices = input_file_indices.split(',')
                        for i in input_file_indices:
                            i = int(i.strip())
                            if i > len(to_extract):
                                print('invalid index try again.')
                                input_file_index = []
                                break

                        tmp = []
                        for index in input_file_indices:
                            index = int(index.strip())
                            print('extracting ', to_extract[index-1])
                            tmp.append(to_extract[index-1])
                        to_extract = tmp
                        break
                    elif len(input_file_indices.strip()) == 1:
                        index = int(input_file_indices.strip())
                        to_extract = [to_extract[index-1]]
                        print('extracting ', to_extract)
                        break
                    elif input_file_indices == '':
                        break
                else:
                    input_file_indices = input_file_indices.strip('e ')
                    if ',' in input_file_indices:
                        input_file_indices = input_file_indices.split(',')
                        for i in input_file_indices:
                            i = int(i.strip())
                            if i > len(to_extract):
                                print('invalid index try again.')
                                input_file_index = []
                                break
                        # values to remove
                        vals = []
                        for index in input_file_indices:
                            index = int(index.strip())
                            print('excluding ', to_extract[index-1])
                            vals.append(to_extract[index-1])
                        for val in vals:
                            to_extract.remove(val)
                        break
                    elif len(input_file_indices.strip()) == 1:
                        index = int(input_file_indices.strip())
                        print('excluding ', to_extract[index-1])
                        to_extract.remove(to_extract[index-1])
                        break
                    elif input_file_indices == '':
                        break

            except:
                print('invalid input, only input correct comma separated indices, or no empty string for all sessions.')
    else:
        print(f'Extracting {to_extract[0]}')

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

    # make the temporary directory if it doesn't already exist
    temp_storage = Path('/tmp/')
    temp_storage.mkdir(parents=True, exist_ok=True)

    suffix = '_{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
    config_store = temp_storage / f'job_config{suffix}.yaml'

    with config_store.open('w') as f:
        yaml.safe_dump(params, f)

    commands = []

    if cluster_type == 'slurm':

        for i, ext in enumerate(to_extract):

            ext = escape_path(ext)
            base_command = 'sbatch -n {:d} --mem={} -p {} -t {} --wrap "'\
                .format(params['cores'], params['memory'], partition, params['wall_time'])
            if prefix is not None:
                base_command += f'{prefix}; '

            if len(params['bg_roi_index']) > 1:
                base_command += 'moseq2-extract find-roi --config-file {} {}; '.format(
                    config_store, ext)

            for roi in params['bg_roi_index']:
                roi_config = deepcopy(params)
                roi_config['bg_roi_index'] = roi
                roi_config_store = escape_path(os.path.join(
                    temp_storage, 'job_config{}_roi{:d}.yaml'.format(suffix, roi)))
                with open(roi_config_store, 'w') as f:
                    yaml.safe_dump(roi_config, f)

                if output_directory is None:
                    base_command += 'moseq2-extract extract --config-file {} --bg-roi-index {:d} {}; '\
                        .format(roi_config_store, roi, ext)
                else:
                    base_command += 'moseq2-extract extract --output-dir {} --config-file {} --bg-roi-index {:d} {}; ' \
                        .format(output_directory, roi_config_store, roi, ext)
                try:
                    extract_command(ext, str(to_extract[i].replace(ext, 'proc/')), roi_config_store, skip=skip_extracted)
                except:
                    print('Unexpected error:', sys.exc_info()[0])
                    print('could not extract', to_extract[i])


            base_command += '"'

            #commands.append(base_command)

    elif cluster_type == 'local':

        for i, ext in enumerate(to_extract):

            base_command = ''

            if prefix is not None:
                base_command += '{}; '.format(prefix)

            if len(params['bg_roi_index']) > 1:
                base_command += 'moseq2-extract find-roi --config-file {} {}; '.format(
                    config_store, ext)

            for roi in params['bg_roi_index']:
                roi_config = deepcopy(params)
                roi_config['bg_roi_index'] = roi
                roi_config_store = os.path.join(
                    temp_storage, 'job_config{}_roi{:d}.yaml'.format(suffix, roi))
                with open(roi_config_store, 'w') as f:
                    yaml.safe_dump(roi_config, f)

                if output_directory is None:
                    base_command += 'moseq2-extract extract --config-file {} --bg-roi-index {:d} {}; '\
                        .format(roi_config_store, roi, ext)
                else:
                    base_command += 'moseq2-extract extract --output-dir {} --config-file {} --bg-roi-index {:d} {}; ' \
                        .format(output_directory, roi_config_store, roi, ext)
                try:
                    extract_command(ext, str(to_extract[i].replace(ext, 'proc/')), roi_config_store, skip=skip_extracted)
                except:
                    print('Unexpected error:', sys.exc_info())
                    print('could not extract', to_extract[i])

            #commands.append(base_command)

    else:
        raise NotImplementedError('Other cluster types not supported')

    print('Extractions Complete.')
    #return commands


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

    with open(os.path.join(input_dir, output_file), 'w') as f:
        yaml.safe_dump(output_dict, f,)

    print('Index file successfully generated.')
    return os.path.join(input_dir, output_file)


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

    # Generate Index file before
    if output_directory is None:
        indexpath = generate_index_command(input_dir, '', 'moseq2-index.yaml', (), False)
    else:
        indexpath = generate_index_command(input_dir, '', os.path.join(output_directory, 'moseq2-index.yaml'), (),
                                           False)
    print(f'Index file path: {indexpath}')


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

    loaded = []
    for _dict, _h5f in tqdm.tqdm_notebook(to_load, desc='Scanning data'):
        try:
            # v0.1.3 introduced a change - acq. metadata now here
            tmp = h5_to_dict(_h5f, '/metadata/acquisition')
        except KeyError:
            # if it doesn't exist it's likely from an older moseq version. Try loading it here
            try:
                tmp = h5_to_dict(_h5f, '/metadata/extraction')
            except KeyError:
                # if all else fails, abandon all hope
                tmp = {}

        # note that everything going into here must be a string (no bytes!)
        tmp = {k: str(v) for k, v in tmp.items()}
        if snake_case:
            tmp = keymap(camel_to_snake, tmp)

        feedback_file = os.path.join(os.path.dirname(_h5f), '..', 'feedback_ts.txt')
        if os.path.exists(feedback_file):
            timestamps = map(int, load_timestamps(feedback_file, 0))
            feedback_status = map(int, load_timestamps(feedback_file, 1))
            _dict['feedback_timestamps'] = list(zip(timestamps, feedback_status))

        _dict['extraction_metadata'] = tmp
        loaded += [(_dict, _h5f)]

    manifest = {}
    fallback = 'session_{:03d}'
    fallback_count = 0

    # you know, bonus internal only stuff for the time being...
    additional_meta = []
    additional_meta.append({
        'filename': 'feedback_ts.txt',
        'var_name': 'realtime_feedback',
        'dtype': np.bool,
    })
    additional_meta.append({
        'filename': 'predictions.txt',
        'var_name': 'realtime_predictions',
        'dtype': np.int,
    })
    additional_meta.append({
        'filename': 'pc_scores.txt',
        'var_name': 'realtime_pc_scores',
        'dtype': np.float32,
    })

    for _dict, _h5f in loaded:
        print_format = '{}_{}'.format(
            format, os.path.splitext(os.path.basename(_h5f))[0])
        if not _dict['extraction_metadata']:
            copy_path = fallback.format(fallback_count)
            fallback_count += 1
        else:
            try:
                copy_path = build_path(_dict['extraction_metadata'], print_format, snake_case=snake_case)
            except:
                copy_path = fallback.format(fallback_count)
                fallback_count += 1
                pass
        # add a bonus dictionary here to be copied to h5 file itself
        manifest[_h5f] = {'copy_path': copy_path, 'yaml_dict': _dict, 'additional_metadata': {}}
        for meta in additional_meta:
            filename = os.path.join(os.path.dirname(_h5f), '..', meta['filename'])
            if os.path.exists(filename):
                try:
                    data, timestamps = load_textdata(filename, dtype=meta['dtype'])
                    manifest[_h5f]['additional_metadata'][meta['var_name']] = {
                        'data': data,
                        'timestamps': timestamps
                    }
                except:
                    print('Did not load timestamps')

    # now the key is the source h5 file and the value is the path to copy to
    for k, v in tqdm.tqdm_notebook(manifest.items(), desc='Copying files'):

        if os.path.exists(os.path.join(output_dir, '{}.h5'.format(v['copy_path']))):
            continue

        basename = os.path.splitext(os.path.basename(k))[0]
        dirname = os.path.dirname(k)

        h5_path = k
        mp4_path = os.path.join(dirname, '{}.mp4'.format(basename))
        # yaml_path = os.path.join(dirname, '{}.yaml'.format(basename))

        if os.path.exists(h5_path):
            new_h5_path = os.path.join(output_dir, '{}.h5'.format(v['copy_path']))
            shutil.copyfile(h5_path, new_h5_path)

        # if we have additional_meta then crack open the h5py and write to a safe place
        if len(v['additional_metadata']) > 0:
            for k2, v2 in v['additional_metadata'].items():
                new_key = '/metadata/misc/{}'.format(k2)
                with h5py.File(new_h5_path, "a") as f:
                    f.create_dataset('{}/data'.format(new_key), data=v2["data"])
                    f.create_dataset('{}/timestamps'.format(new_key), data=v2["timestamps"])

        if os.path.exists(mp4_path):
            shutil.copyfile(mp4_path, os.path.join(
                output_dir, '{}.mp4'.format(v['copy_path'])))

        v['yaml_dict'].pop('extraction_metadata', None)
        with open('{}.yaml'.format(os.path.join(output_dir, v['copy_path'])), 'w') as f:
            yaml.safe_dump(v['yaml_dict'], f)

    print('Results successfully aggregated in', output_dir)


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
                #return data_dir, 0

    # generate sample metadata json for each session that is missing one

    sample_meta = {'SubjectName': 'default', 'SessionName':'default',
                   'NidaqChannels': 0, 'NidaqSamplingRate':0.0, 'DepthResolution': [512, 424],
                   'ColorDataType':"Byte[]", "StartTime":""}
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

    print('Getting background...')
    bground_im = get_bground_im_file(input_file)
    write_image(os.path.join(output_dir, 'bground.tiff'), bground_im, scale=True)

    first_frame = load_movie_data(input_file, 0)
    write_image(os.path.join(output_dir, 'first_frame.tiff'), first_frame, scale=True,
                scale_factor=config_data['bg_roi_depth_range'])

    print('Getting roi...')

    strel_dilate = select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_dilate']))

    rois, _, _, _, _, _ = get_roi(bground_im,
                                  strel_dilate=strel_dilate,
                                  dilate_iters=config_data.get('dilate_iterations', 1),
                                  weights=config_data['bg_roi_weights'],
                                  noise_tolerance=config_data.get('noise_tolerance', 30),
                                  depth_range=config_data['bg_roi_depth_range'],
                                  gradient_filter=config_data['bg_roi_gradient_filter'],
                                  gradient_threshold=config_data['bg_roi_gradient_threshold'],
                                  gradient_kernel=config_data['bg_roi_gradient_kernel'],
                                  fill_holes=config_data['bg_roi_fill_holes'], gui=True)

    if config_data['bg_sort_roi_by_position']:
        rois = rois[:config_data['bg_sort_roi_by_position_max_rois']]
        rois = [rois[i] for i in np.argsort([np.nonzero(roi)[0].mean() for roi in rois])]

    bg_roi_index = [idx for idx in bg_roi_index if idx in range(len(rois))]
    for idx in bg_roi_index:
        roi_filename = 'roi_{:02d}.tiff'.format(idx)
        write_image(os.path.join(output_dir, roi_filename),
                    rois[idx], scale=True, dtype='uint8')

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

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    with open(config_file, 'w') as g:
        yaml.safe_dump(config_data, g)
    g.close()

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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = 'results_{:02d}'.format(config_data['bg_roi_index'])
    status_filename = os.path.join(output_dir, '{}.testyaml'.format(output_filename))

    if os.path.exists(status_filename):
        overwrite = input('Press ENTER to overwrite your previous extraction, else to end the process.')
        if overwrite != '':
            raise RuntimeError("Already found a status file in {}, delete and try again".format(status_filename))

    with open(status_filename, 'w') as f:
        yaml.safe_dump(status_dict, f)

    bg_roi_file = input_file

    # get the background and roi, which will be used across all batches
    print('Getting background...')
    bground_im = get_bground_im_file(bg_roi_file, tar_object=tar)
    if not config_data['use_plane_bground']:
        write_image(os.path.join(output_dir, 'bground.tiff'), bground_im, scale=True)

    first_frame = load_movie_data(input_file, 0, tar_object=tar)
    write_image(os.path.join(output_dir, 'first_frame.tiff'), first_frame, scale=True,
                scale_factor=config_data['bg_roi_depth_range'])

    roi_filename = 'roi_{:02d}.tiff'.format(config_data['bg_roi_index'])

    strel_dilate = select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_dilate']))
    strel_tail = select_strel(config_data['tail_filter_shape'], tuple(config_data['tail_filter_size']))
    strel_min = select_strel(config_data['cable_filter_shape'], tuple(config_data['cable_filter_size']))

    print('Getting roi...')
    rois, plane, _, _, _, _ = get_roi(bground_im,
                                      strel_dilate=strel_dilate,
                                      dilate_iters=config_data.get('dilate_iterations', 1),
                                      weights=config_data['bg_roi_weights'],
                                      depth_range=config_data['bg_roi_depth_range'],
                                      gradient_filter=config_data['bg_roi_gradient_filter'],
                                      gradient_threshold=config_data['bg_roi_gradient_threshold'],
                                      noise_tolerance=config_data.get('noise_tolerance', 30),
                                      gradient_kernel=config_data['bg_roi_gradient_kernel'],
                                      fill_holes=config_data['bg_roi_fill_holes'],
                                      gui=True)


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

    if config_data.get('detected_true_depth', 'auto') == 'auto':
        true_depth = np.median(bground_im[roi > 0])
    else:
        true_depth = config_data['detected_true_depth']

    print('Detected true depth: {}'.format(true_depth))

    if config_data['dilate_iterations'] > 1:
        print('Dilating background')

        bground_im = graduate_dilated_wall_area(bground_im, config_data, strel_dilate, true_depth, output_dir)

    # farm out the batches and write to an hdf5 file
    with h5py.File(os.path.join(output_dir, '{}.testh5'.format(output_filename)), 'w') as f:
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

        #save_dict_contents_to_h5(f, status_dict['parameters'], 'metadata/extraction/parameters', click_param_annot(extract))

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

        for i, frame_range in enumerate(tqdm.tqdm_notebook(frame_batches, desc='Processing batches')):
            raw_frames = load_movie_data(input_file, [f + first_frame_idx for f in frame_range], tar_object=tar)
            raw_frames = bground_im-raw_frames
            # raw_frames[np.logical_or(raw_frames < min_height, raw_frames > max_height)] = 0
            raw_frames[raw_frames < config_data['min_height']] = 0
            if config_data['dilate_iterations'] == 1:
                raw_frames[raw_frames > config_data['max_height']] = config_data['max_height']
            else:
                raw_frames[raw_frames > config_data['max_height']] = 0

            raw_frames = raw_frames.astype(config_data['frame_dtype'])
            raw_frames = apply_roi(raw_frames, roi)

            results = extract_chunk(raw_frames,
                                    use_em_tracker=config_data['use_tracking_model'],
                                    strel_tail=strel_tail,
                                    strel_min=strel_min,
                                    use_cc=config_data.get('use_cc', False),
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
        yaml.safe_dump(status_dict, f)

    print(f'Sample extraction of {str(nframes)} frames completed successfully in {output_dir}.')
    return output_dir

def extract_command(input_file, output_dir, config_file, skip=False):
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

    if skip == True:
        if os.path.exists(os.path.join(output_dir, 'done.txt')):
            return

    with open(status_filename, 'w') as f:
        yaml.safe_dump(status_dict, f)

    bg_roi_file = input_file

    print('Getting background...')
    bground_im = get_bground_im_file(bg_roi_file, tar_object=tar)
    if not config_data['use_plane_bground']:
        write_image(os.path.join(output_dir, 'bground.tiff'), bground_im, scale=True)

    first_frame = load_movie_data(input_file, 0, tar_object=tar)
    write_image(os.path.join(output_dir, 'first_frame.tiff'), first_frame, scale=True,
                scale_factor=config_data['bg_roi_depth_range'])

    roi_filename = 'roi_{:02d}.tiff'.format(config_data['bg_roi_index'])

    strel_dilate = select_strel((config_data['bg_roi_shape'], config_data['bg_roi_dilate']))
    strel_tail = select_strel((config_data['tail_filter_shape'], config_data['tail_filter_size']))
    strel_min = select_strel((config_data['cable_filter_shape'], config_data['cable_filter_size']))

    print('Getting roi...')
    rois, plane, _, _, _, _ = get_roi(bground_im,
                                      strel_dilate=strel_dilate,
                                      dilate_iters=config_data.get('dilate_iterations', 1),
                                      weights=config_data['bg_roi_weights'],
                                      noise_tolerance=config_data.get('noise_tolerance', 30),
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

    if config_data['dilate_iterations'] > 1:
        print('Dilating background')
        bground_im = graduate_dilated_wall_area(bground_im, config_data, strel_dilate, true_depth, output_dir)


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

        for i, frame_range in enumerate(tqdm.tqdm_notebook(frame_batches, desc='Processing batches')):
            raw_frames = load_movie_data(input_file, [f + first_frame_idx for f in frame_range], tar_object=tar)
            raw_frames = bground_im-raw_frames
            # raw_frames[np.logical_or(raw_frames < min_height, raw_frames > max_height)] = 0
            raw_frames[raw_frames < config_data['min_height']] = 0
            if config_data['dilate_iterations'] == 1:
                raw_frames[raw_frames > config_data['max_height']] = config_data['max_height']
            else:
                raw_frames[raw_frames > config_data['max_height']] = 0
            raw_frames = raw_frames.astype(config_data['frame_dtype'])
            raw_frames = apply_roi(raw_frames, roi)

            results = extract_chunk(raw_frames,
                                    use_em_tracker=config_data['use_tracking_model'],
                                    strel_tail=strel_tail,
                                    strel_min=strel_min,
                                    use_cc=config_data.get('use_cc', False),
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
                                    progress_bar=False,
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

    with open(os.path.join(output_dir, 'done.txt'), 'w') as f:
        f.write('done')

    with open(status_filename, 'w') as f:
        yaml.safe_dump(status_dict, f)

    return 'Extraction completed.'