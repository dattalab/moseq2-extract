import pytest
import os
import ruamel.yaml as yaml
import click
import numpy.testing as npt
import numpy as np
import cv2
import time
from click.testing import CliRunner
import shutil
from moseq2_extract.cli import find_roi, extract, download_flip_file, generate_config, copy_slice


@pytest.fixture(scope='function')
def temp_dir(tmpdir):
    f = tmpdir.mkdir('test_dir')
    return str(f)

def test_copy_slice():

    temp_dir = 'tests/test_video_files/'
    data_path = os.path.join(temp_dir, 'extract_vid.avi')
    output_path = 'tests/test_video_files/extracted_copy/output.avi'

    slice_params = ['--output-file', output_path,
                    '--chunk-size', 2000,
                    '--copy-slice', '0', '1000',
                    '--fps', 30,
                    '--delete', False,
                    '--threads', 3,
                    data_path]

    runner = CliRunner()
    result = runner.invoke(copy_slice, slice_params)

    assert (os.path.exists(output_path) == True)
    if True:
        os.remove(output_path)

    assert (result.exit_code == 0)

def test_extract():

    extract_dir = 'tests/test_video_files/'
    flip_path = 'tests/test_flip_classifiers/flip_test1.pkl'
    output_path = 'extracted/'
    data_path = os.path.join(extract_dir, 'test_raw.dat')
    param_set = [data_path,
                 '--output-dir', output_path,
                 #'--config_file', None, ## IS OPTIONAL FLAG
                 '--crop-size', 80, 80,
                 '--bg-roi-dilate', 10, 10,
                 '--bg-roi-shape', 'ellipse',
                 '--bg-roi-index', 0,
                 '--bg-roi-weights', 1, .1, 1,
                 '--bg-roi-depth-range', 650, 750,
                 '--bg-roi-gradient-kernel', 7,
                 '--bg-roi-fill-holes', True,
                 '--min-height', 10,
                 '--max-height', 100,
                 '--fps', 30,
                 '--flip-classifier', flip_path,
                 '--flip-classifier-smoothing', 51,
                 '--use-tracking-model', False,
                 '--tracking-model-ll-threshold', -100,
                 '--tracking-model-ll-clip', -100,
                 '--tracking-model-mask-threshold', -16,
                 '--tracking-model-segment', True,
                 '--tracking-model-init', 'raw',
                 '--cable-filter-iters', 0,
                 '--cable-filter-size', 5, 5,
                 '--cable-filter-shape', 'rectangle',
                 '--tail-filter-size', 9, 9,
                 '--tail-filter-iters', 1,
                 '--tail-filter-shape', 'ellipse',
                 '--spatial-filter-size', 3,
                 '--temporal-filter-size', 0,
                 '--chunk-size', 1200,
                 '--chunk-overlap', 0,
                 '--write-movie', True,
                 '--use-plane-bground', # FLAG
                 '--frame-dtype', 'uint8',
                 '--angle-hampel-span', 0,
                 '--angle-hampel-sig', 3,
                 '--centroid-hampel-span', 0,
                 '--centroid-hampel-sig', 3,
                 '--model-smoothing-clips', 0, 0,
                 '--frame-trim',0 , 0,
                 '--compress', False,
                 '--compress-chunk-size', 3000,
                 '--compress-threads', 3]

    edge_size = 40
    points = np.arange(-edge_size, edge_size)
    sig1 = 10
    sig2 = 20

    kernel = np.exp(-(points**2.0) / (2.0 * sig1**2.0))
    kernel2 = np.exp(-(points**2.0) / (2.0 * sig2**2.0))

    kernel_full = np.outer(kernel, kernel2)
    kernel_full /= np.max(kernel_full)
    kernel_full *= 50

    fake_mouse = kernel_full
    fake_mouse[fake_mouse < 5] = 0

    tmp_image = np.ones((424, 512), dtype='int16') * 1000
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2

    # put a mouse on top of a disk

    roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (300, 300)).astype('int16') * 300
    roi_dims = np.array(roi.shape) // 2

    tmp_image[center[0]-roi_dims[0]:center[0]+roi_dims[0],
              center[1]-roi_dims[1]:center[1]+roi_dims[1]] =\
        tmp_image[center[0]-roi_dims[0]:center[0]+roi_dims[0],
                  center[1]-roi_dims[1]:center[1]+roi_dims[1]]-roi

    tmp_image[center[0]-mouse_dims[0]:center[0]+mouse_dims[0],
              center[1]-mouse_dims[1]:center[1]+mouse_dims[1]] =\
        tmp_image[center[0]-mouse_dims[0]:center[0]+mouse_dims[0],
                  center[1]-mouse_dims[1]:center[1]+mouse_dims[1]]-fake_mouse

    fake_movie = np.tile(tmp_image, (20, 1, 1))
    fake_movie.tofile(data_path)

    runner = CliRunner()
    result = runner.invoke(extract, param_set,
                           catch_exceptions=False)

    assert (os.path.exists(extract_dir+'extracted/') == True)
    shutil.rmtree(extract_dir+'extracted/')
    assert(result.exit_code == 0)

def test_extract_trim(temp_dir):

    data_path = os.path.join(temp_dir, 'test_raw.dat')
    edge_size = 40
    points = np.arange(-edge_size, edge_size)
    sig1 = 10
    sig2 = 20

    kernel = np.exp(-(points**2.0) / (2.0 * sig1**2.0))
    kernel2 = np.exp(-(points**2.0) / (2.0 * sig2**2.0))

    kernel_full = np.outer(kernel, kernel2)
    kernel_full /= np.max(kernel_full)
    kernel_full *= 50

    fake_mouse = kernel_full
    fake_mouse[fake_mouse < 5] = 0

    tmp_image = np.ones((424, 512), dtype='int16') * 1000
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2

    # put a mouse on top of a disk

    roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (300, 300)).astype('int16') * 300
    roi_dims = np.array(roi.shape) // 2

    tmp_image[center[0]-roi_dims[0]:center[0]+roi_dims[0],
              center[1]-roi_dims[1]:center[1]+roi_dims[1]] =\
        tmp_image[center[0]-roi_dims[0]:center[0]+roi_dims[0],
                  center[1]-roi_dims[1]:center[1]+roi_dims[1]]-roi

    tmp_image[center[0]-mouse_dims[0]:center[0]+mouse_dims[0],
              center[1]-mouse_dims[1]:center[1]+mouse_dims[1]] =\
        tmp_image[center[0]-mouse_dims[0]:center[0]+mouse_dims[0],
                  center[1]-mouse_dims[1]:center[1]+mouse_dims[1]]-fake_mouse

    fake_movie = np.tile(tmp_image, (300, 1, 1))
    fake_movie.tofile(data_path)

    runner = CliRunner()
    result = runner.invoke(extract, [data_path,
                                     '--output-dir', temp_dir,
                                     '--frame-trim', 20, 50],
                           catch_exceptions=False)

    assert (os.path.exists(temp_dir) == True)

    assert(result.exit_code == 0)


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
        pytest.fail('raw was not deleted')
    if 'test_raw.avi' not in files[0]:
        print(files)
        pytest.fail('avi file not found')

    assert (os.path.exists("tests/test_video_files/test_raw.avi") == True)

    os.remove('tests/test_video_files/test_raw.avi')


def test_find_roi():
    temp_dir = 'tests/test_video_files/roi_output/'
    data_path = os.path.join(temp_dir, 'test_raw.dat')

    roi_params = [data_path,
                  '--output-dir', temp_dir,
                  '--bg-roi-dilate', 10, 10,
                  '--bg-roi-shape', 'ellipse',
                  '--bg-roi-index', 0,
                  '--bg-roi-weights', 1, .1, 1,
                  '--bg-roi-depth-range', 650, 750,
                  '--bg-roi-gradient-filter', False,
                  '--bg-roi-gradient-threshold', 3000,
                  '--bg-roi-gradient-kernel', 7,
                  '--bg-roi-fill-holes', True,
                  '--bg-sort-roi-by-position', False,
                  '--bg-sort-roi-by-position-max-rois', 2,
                  '--use-plane-bground', False]
    edge_size = 40
    points = np.arange(-edge_size, edge_size)
    sig1 = 10
    sig2 = 20

    kernel = np.exp(-(points**2.0) / (2.0 * sig1**2.0))
    kernel2 = np.exp(-(points**2.0) / (2.0 * sig2**2.0))

    kernel_full = np.outer(kernel, kernel2)
    kernel_full /= np.max(kernel_full)
    kernel_full *= 50

    fake_mouse = kernel_full
    fake_mouse[fake_mouse < 5] = 0

    tmp_image = np.ones((424, 512), dtype='int16') * 1000
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2

    # put a mouse on top of a disk

    roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (300, 300)).astype('int16') * 300
    roi_dims = np.array(roi.shape) // 2

    tmp_image[center[0]-roi_dims[0]:center[0]+roi_dims[0],
              center[1]-roi_dims[1]:center[1]+roi_dims[1]] =\
        tmp_image[center[0]-roi_dims[0]:center[0]+roi_dims[0],
                  center[1]-roi_dims[1]:center[1]+roi_dims[1]]-roi

    tmp_image[center[0]-mouse_dims[0]:center[0]+mouse_dims[0],
              center[1]-mouse_dims[1]:center[1]+mouse_dims[1]] =\
        tmp_image[center[0]-mouse_dims[0]:center[0]+mouse_dims[0],
                  center[1]-mouse_dims[1]:center[1]+mouse_dims[1]]-fake_mouse

    fake_movie = np.tile(tmp_image, (20, 1, 1))
    fake_movie.tofile(data_path)

    runner = CliRunner()
    result = runner.invoke(find_roi, roi_params)

    assert (os.path.exists(temp_dir) == True)
    shutil.rmtree(temp_dir)
    assert(result.exit_code == 0)


def test_download_flip_file(temp_dir):

    runner = CliRunner()
    result = runner.invoke(download_flip_file, ['--output-dir', temp_dir], input='0\n')

    assert(os.path.exists(temp_dir) == True)
    assert(result.exit_code == 0)


def test_generate_config(temp_dir):

    temp_path = os.path.join(temp_dir, 'config.yaml')
    runner = CliRunner()
    result = runner.invoke(generate_config, ['--output-file', temp_path])
    yaml_data = yaml.load(temp_path, Loader=yaml.RoundTripLoader)
    tmp = extract.params
    params = [param for param in tmp if type(tmp) is click.core.Option]

    assert(os.path.exists(temp_path) == True)
    assert(result.exit_code == 0)

    for param in params:
        npt.assert_equal(yaml_data[param.human_readable_name], param.default)


# def test_extract_em(temp_dir):
#
#     data_path = os.path.join(temp_dir, 'test_vid.dat')
#     edge_size = 40
#     points = np.arange(-edge_size, edge_size)
#     sig1 = 10
#     sig2 = 20
#
#     kernel = np.exp(-(points**2.0) / (2.0 * sig1**2.0))
#     kernel2 = np.exp(-(points**2.0) / (2.0 * sig2**2.0))
#
#     kernel_full = np.outer(kernel, kernel2)
#     kernel_full /= np.max(kernel_full)
#     kernel_full *= 50
#
#     fake_mouse = kernel_full
#     fake_mouse[fake_mouse < 5] = 0
#
#     tmp_image = np.ones((424, 512), dtype='int16') * 1000
#     center = np.array(tmp_image.shape) // 2
#
#     mouse_dims = np.array(fake_mouse.shape) // 2
#
#     # put a mouse on top of a disk
#
#     roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (300, 300)).astype('int16') * 300
#     roi_dims = np.array(roi.shape) // 2
#
#     tmp_image[center[0]-roi_dims[0]:center[0]+roi_dims[0],
#               center[1]-roi_dims[1]:center[1]+roi_dims[1]] =\
#         tmp_image[center[0]-roi_dims[0]:center[0]+roi_dims[0],
#                   center[1]-roi_dims[1]:center[1]+roi_dims[1]]-roi
#
#     tmp_image[center[0]-mouse_dims[0]:center[0]+mouse_dims[0],
#               center[1]-mouse_dims[1]:center[1]+mouse_dims[1]] =\
#         tmp_image[center[0]-mouse_dims[0]:center[0]+mouse_dims[0],
#                   center[1]-mouse_dims[1]:center[1]+mouse_dims[1]]-fake_mouse
#
#     fake_movie = np.tile(tmp_image, (300, 1, 1))
#     fake_movie.tofile(data_path)
#
#     runner = CliRunner()
#     result = runner.invoke(extract, [data_path,
#                                      '--output-dir', temp_dir,
#                                      '--use-tracking-model', True],
#                            catch_exceptions=True)
#     print(result.output)
#     assert(result.exit_code == 0)
