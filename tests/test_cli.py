import pytest
import os
import ruamel.yaml as yaml
import click
import numpy.testing as npt
import numpy as np
import cv2
from click.testing import CliRunner
from moseq2_extract.cli import find_roi, extract, download_flip_file, generate_config


@pytest.fixture(scope='function')
def temp_dir(tmpdir):
    f = tmpdir.mkdir('test_dir')
    return str(f)


def test_extract(temp_dir):

    data_path = os.path.join(temp_dir, 'depth.dat')
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
    result = runner.invoke(extract, [data_path, '--output-dir', temp_dir])

    assert(result.exit_code == 0)


def test_find_roi(temp_dir):

    data_path = os.path.join(temp_dir, 'depth.dat')
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
    result = runner.invoke(find_roi, [data_path, '--output-dir', temp_dir])

    assert(result.exit_code == 0)


def test_download_flip_file(temp_dir):

    runner = CliRunner()
    result = runner.invoke(download_flip_file, ['--output-dir', temp_dir], input='0\n')
    assert(result.exit_code == 0)


def test_generate_config(temp_dir):

    temp_path = os.path.join(temp_dir, 'config.yaml')
    runner = CliRunner()
    result = runner.invoke(generate_config, ['--output-file', temp_path])
    yaml_data = yaml.load(temp_path, Loader=yaml.RoundTripLoader)
    tmp = extract.params
    params = [param for param in tmp if type(tmp) is click.core.Option]

    assert(result.exit_code == 0)

    for param in params:
        npt.assert_equal(yaml_data[param.human_readable_name], param.default)
