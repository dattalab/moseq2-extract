import sys

import pytest
import unittest
import os
import numpy as np
import numpy.testing as npt
import json
import cv2
import time
import click
import h5py

from moseq2_extract.cli import find_roi
from moseq2_extract.util import gen_batch_sequence, load_metadata, load_timestamps,\
    select_strel, command_with_config, scalar_attributes, save_dict_contents_to_h5,\
    click_param_annot


@pytest.fixture(scope='function')
def temp_dir(tmpdir):
    f = tmpdir.mkdir('test_dir')
    return str(f)

def test_strided_app():
    # original params: a, L, S
    # Window len = L, Stride len/stepsize = S
    a = np.asarray([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    L = 5
    S = 3

    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]

    mock_result = np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))
    test_result = [[1, 2, 3, 4, 5],
     [4, 5, 6, 7, 8],
     [7, 8, 9, 10, 11]]

    assert all([c == d for a, b in zip(mock_result, test_result) for c, d in zip(a, b)])


def test_convert_raw_to_avi_function():
    # original params: input_file, chunk_size=2000, fps=30, delete=False, threads=3
    input_file = 'tests/test_video_files/test_raw.dat'
    chunk_size = 2000
    fps = 30
    delete = False
    threads = 3

    new_file = '{}.avi'.format(os.path.splitext(input_file)[0])

    # turn into os system call...
    use_kwargs = {
        'output-file': new_file,
        'chunk-size': chunk_size,
        'fps': fps,
        'threads': threads
    }
    use_flags = {
        'delete': delete
    }
    base_command = 'moseq2-extract convert-raw-to-avi {}'.format(input_file)
    for k, v in use_kwargs.items():
        base_command += ' --{} {}'.format(k, v)
    for k, v in use_flags.items():
        if v:
            base_command += ' --{}'.format(k)

    print(base_command)
    print('\n')

    os.system(base_command)
    time.sleep(14) # waiting for file to save to desired test dir

    files = [os.listdir('tests/test_video_files')]
    if ('test_raw.dat' in files[0]) and (delete):
        print(files)
        #pytest.fail('raw was not deleted')
    if 'test_vid.avi' not in files[0]:
        print(files)
        pytest.fail('avi file not found')

    os.remove('tests/test_video_files/test_vid.avi')


def test_gen_batch_sequence():

    tmp_list = [range(0, 10),
                range(5, 15),
                range(10, 20),
                range(15, 25)]

    gen_list = list(gen_batch_sequence(25, 10, 5))

    assert(gen_list == tmp_list)


def test_load_timestamps(temp_dir):

    tmp_timestamps = np.arange(0, 5, .05)
    txt_file = os.path.join(temp_dir, 'test_timestamps.txt')

    with open(txt_file, 'w') as f:
        for timestamp in tmp_timestamps:
            print('{}'.format(str(timestamp)), file=f)

    loaded_timestamps = load_timestamps(txt_file)
    npt.assert_almost_equal(loaded_timestamps, tmp_timestamps, 10)


def test_load_metadata(temp_dir):

    tmp_dict = {
        'test': 'test2'
    }

    json_file = os.path.join(temp_dir, 'test_json.json')
    with open(json_file, 'w') as f:
        json.dump(tmp_dict, f)

    loaded_dict = load_metadata(json_file)

    assert(loaded_dict == tmp_dict)


def test_select_strel():

    strel = select_strel('ellipse', size=(9, 9))
    npt.assert_equal(strel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    strel = select_strel('rectangle', size=(9, 9))
    npt.assert_equal(strel, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))

    strel = select_strel('sdfdfsf', size=(9, 9))
    npt.assert_equal(strel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


def test_scalar_attributes():

    dct = scalar_attributes()

    assert(dct is not None)


def test_save_dict_contents_to_h5(temp_dir):

    tmp_dic = {
        'subdict': {
            'sd_tuple': (0,1),
            'sd_string': 'quick brown fox',
            'sd_integer': 1,
            'sd_float': 1.0,
            'sd_bool': False,
            'sd_list': [1,2,3],
        },
        'tuple': (0,1),
        'string': 'quick brown fox',
        'integer': 1,
        'float': 1.0,
        'bool': False,
        'list': [1,2,3],
    }
    root_path = '/myroot'
    fpath = os.path.join(temp_dir, 'test.h5')
    f = h5py.File(fpath, 'w')
    save_dict_contents_to_h5(f, tmp_dic, root_path)
    f.close()

    def h5_to_dict(h5file, path):
        ans = {}
        if not path.endswith('/'):
            path = path + '/'
        for key, item in h5file[path].items():
            if type(item) is h5py.Dataset:
                ans[key] = item[()]
            elif type(item) is h5py.Group:
                ans[key] = h5_to_dict(h5file, path + key + '/')
        return ans

    result = h5_to_dict(h5py.File(fpath, 'r'), root_path)
    npt.assert_equal(result, tmp_dic)

def test_click_param_annot():
    ref_dict = {
        'bg_roi_dilate': 'Size of strel to dilate roi',
        'bg_roi_shape': 'Shape to use to dilate roi (ellipse or rect)',
        'bg_roi_index': 'Index of roi to use',
        'bg_roi_weights': 'ROI feature weighting (area, extent, dist)',
        'bg_roi_depth_range': 'Range to search for floor of arena (in mm)',
        'bg_roi_gradient_filter': 'Exclude walls with gradient filtering',
        'bg_roi_gradient_threshold': 'Gradient must be < this to include points',
        'bg_roi_gradient_kernel': 'Kernel size for Sobel gradient filtering',
        'bg_sort_roi_by_position': 'Sort ROIs by position',
        'bg_sort_roi_by_position_max_rois': 'Max original ROIs to sort by position',
        'bg_roi_fill_holes': 'Fill holes in ROI',
        'output_dir': 'Output directory',
        'use_plane_bground': 'Use plane fit for background',
        'config_file': None
    }
    test_dict = click_param_annot(find_roi)
    npt.assert_equal(ref_dict, test_dict)
