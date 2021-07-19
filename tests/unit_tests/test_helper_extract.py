import os
import cv2
import h5py
import uuid
import shutil
import numpy as np
from copy import deepcopy
import ruamel.yaml as yaml
from unittest import TestCase
from moseq2_extract.io.image import read_image
from moseq2_extract.helpers.data import create_extract_h5
from ..integration_tests.test_cli import write_fake_movie
from moseq2_extract.gui import generate_config_command, download_flip_command
from moseq2_extract.util import scalar_attributes, gen_batch_sequence, load_metadata
from moseq2_extract.helpers.extract import run_local_extract, process_extract_batches, write_extracted_chunk_to_h5

class TestHelperExtract(TestCase):

    def test_write_extracted_chunk_to_h5(self):

        output_dir = 'data/'
        output_filename = 'test_out'
        offset = 0
        frame_range = range(0, 100)
        scalars = ['speed']
        config_data = {'flip_classifier': False}
        results = {'depth_frames': np.zeros((100, 10, 10)),
                   'mask_frames': np.zeros((100, 10, 10)),
                   'scalars': {'speed': np.ones((100, 1))}
                   }

        out_file = os.path.join(output_dir, f'{output_filename}.h5')

        with h5py.File(out_file, 'w') as f:
            f.create_dataset(f'scalars/speed', (100, 1), 'float32', compression='gzip')
            f.create_dataset(f'frames', (100, 10, 10), 'float32', compression='gzip')
            f.create_dataset(f'frames_mask', (100, 10, 10), 'float32', compression='gzip')

            write_extracted_chunk_to_h5(f, results, config_data, scalars, frame_range, offset)

        assert os.path.exists(out_file)
        os.remove(out_file)

    def test_process_extract_batches(self):

        output_dir = 'data/'
        config_file = 'data/config.yaml'
        metadata_path = 'data/metadata.json'
        output_filename = 'test_out'

        bground_im = read_image(os.path.join(output_dir, 'tiffs/', 'bground_bucket.tiff'), scale=True)
        roi = read_image(os.path.join(output_dir, 'tiffs/', 'roi_bucket_01.tiff'), scale=True)
        first_frame = np.zeros(roi.shape)
        true_depth = 773.0
        first_frame_idx = 0
        nframes = 20

        flip_file = 'data/flip/flip_classifier_k2_c57_10to13weeks.pkl'
        if not os.path.isfile(flip_file):
            download_flip_command('data/flip/')

        assert os.path.isfile(flip_file), 'flip file was not correctly downloaded'

        acquisition_metadata = load_metadata(metadata_path)

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        config_data['flip_classifier'] = flip_file
        config_data['true_depth'] = true_depth
        config_data['tar'] = False

        status_dict = {
            'parameters': deepcopy(config_data),
            'complete': False,
            'skip': False,
            'uuid': str(uuid.uuid4()),
            'metadata': ''
        }

        frame_batches = list(gen_batch_sequence(nframes, config_data['chunk_size'], config_data['chunk_overlap']))

        str_els = {
            'strel_tail': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            'strel_min': cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        }

        scalars_attrs = scalar_attributes()
        scalars = list(scalars_attrs.keys())

        # writing a file to test following pipeline
        data_path = 'data/unit_test/'
        data_file = f'{data_path}depth.dat'
        if not os.path.isdir('data/unit_test/'):
            os.makedirs('data/unit_test/')

        write_fake_movie(data_file)
        assert os.path.isfile(data_file)

        with h5py.File(os.path.join(output_dir, f'{output_filename}.h5'), 'w') as g:
            create_extract_h5(g, acquisition_metadata, config_data, status_dict, scalars_attrs, nframes,
                              roi, bground_im, first_frame, None)

            process_extract_batches(data_file, config_data, bground_im, roi, frame_batches,
                                    first_frame_idx, str_els, os.path.join(output_dir, output_filename+'.mp4'),
                                    scalars=scalars, h5_file=g)

        assert os.path.exists(os.path.join(output_dir, f'{output_filename}.h5'))
        os.remove(os.path.join(output_dir, f'{output_filename}.h5'))
        assert os.path.exists(os.path.join(output_dir, f'{output_filename}.mp4'))
        os.remove(os.path.join(output_dir, f'{output_filename}.mp4'))
        shutil.rmtree(data_path)
        shutil.rmtree('data/flip/')

    def test_run_local_extract(self):

        config_path = 'data/test_local_ex_config.yaml'
        generate_config_command(config_path)

        # writing a file to test following pipeline
        data_dir = 'data/test_local_session/'
        data_path = f'{data_dir}depth.dat'
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        write_fake_movie(data_path)
        assert os.path.isfile(data_path), "fake movie was not written correctly"

        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)

        prefix = ''

        run_local_extract([str(data_path)], params, prefix)
        os.remove(config_path)
        shutil.rmtree(data_dir)