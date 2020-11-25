import os
import cv2
import glob
import click
import shutil
import numpy as np
import ruamel.yaml as yaml
import numpy.testing as npt
from unittest import TestCase
from click.testing import CliRunner
from moseq2_extract.util import read_yaml
from moseq2_extract.cli import find_roi, extract, download_flip_file, generate_config, \
    convert_raw_to_avi, copy_slice, generate_index, aggregate_extract_results

def write_fake_movie(data_path):
    edge_size = 40
    points = np.arange(-edge_size, edge_size)
    sig1 = 10
    sig2 = 20

    kernel = np.exp(-(points ** 2.0) / (2.0 * sig1 ** 2.0))
    kernel2 = np.exp(-(points ** 2.0) / (2.0 * sig2 ** 2.0))

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

    tmp_image[center[0] - roi_dims[0]:center[0] + roi_dims[0],
    center[1] - roi_dims[1]:center[1] + roi_dims[1]] = \
        tmp_image[center[0] - roi_dims[0]:center[0] + roi_dims[0],
        center[1] - roi_dims[1]:center[1] + roi_dims[1]] - roi

    tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
    center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = \
        tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
        center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] - fake_mouse

    fake_movie = np.tile(tmp_image, (20, 1, 1))
    fake_movie.tofile(data_path)


class CLITests(TestCase):

    def test_aggregate_extract_results(self):

        input_dir = 'data/'
        output_dir = 'data/aggregate_results'

        params = ['-i', input_dir,
                  '-o', output_dir]

        runner = CliRunner()
        result = runner.invoke(aggregate_extract_results, params, catch_exceptions=False)

        assert (result.exit_code == 0), "CLI command did not successfully complete"
        assert os.path.isdir(output_dir), "aggregate results directory was not created"
        assert len(os.listdir(output_dir)) == 2
        shutil.rmtree(output_dir)

    def test_generate_index(self):
        input_dir = 'data/'
        output_file = 'data/moseq2-index.yaml'

        params = ['-i', input_dir,
                  '-o', output_file]

        runner = CliRunner()
        result = runner.invoke(generate_index, params, catch_exceptions=False)

        assert (result.exit_code == 0), "CLI command did not successfully complete"
        assert os.path.isfile(output_file)
        os.remove(output_file)


    def test_extract(self):

        data_path = 'data/extract_test_depth.dat'
        config_file = 'data/config.yaml'

        config_data = read_yaml(config_file)
        config_data['flip_classifier'] = None

        with open(config_file, 'w+') as f:
            yaml.safe_dump(config_data, f)

        write_fake_movie(data_path)
        assert os.path.isfile(data_path), "fake movie was not written"

        runner = CliRunner()
        result = runner.invoke(extract, [data_path, '--output-dir', 'test_out/', '--compute-raw-scalars',
                                         '--config-file', config_file,
                                         '--use-tracking-model', True],
                               catch_exceptions=False)

        assert(result.exit_code == 0), "CLI command did not successfully complete"
        shutil.rmtree('data/test_out/')
        os.remove(data_path)


    def test_find_roi(self):

        data_path = 'data/roi_test_depth.dat'
        out_path = 'out/'
        output_dir = 'data/out/'

        write_fake_movie(data_path)

        runner = CliRunner()
        result = runner.invoke(find_roi, [data_path, '--output-dir', out_path])

        assert(result.exit_code == 0), "CLI command did not successfully complete"
        assert len(glob.glob(output_dir+'*.tiff')) == 3, \
            "ROI files were not generated in the correct directory"

        shutil.rmtree(output_dir)
        os.remove(data_path)

    def test_download_flip_file(self):

        data_path = 'data/config.yaml'
        out_path = 'data/flip/'

        runner = CliRunner()
        result = runner.invoke(download_flip_file, [data_path, '--output-dir', out_path], input='0\n')
        assert (result.exit_code == 0), "CLI command did not complete successfully"
        assert len(glob.glob('data/flip/*.pkl')) > 0, "Flip file was not downloaded correctly"

        shutil.rmtree(out_path)

    def test_generate_config(self):

        data_path = 'data/test_config.yaml'

        runner = CliRunner()
        result = runner.invoke(generate_config, ['--output-file', data_path])
        yaml_data = yaml.load('data/', Loader=yaml.RoundTripLoader)
        temp_p = extract.params
        params = [param for param in temp_p if type(temp_p) is click.core.Option]

        for param in params:
            npt.assert_equal(yaml_data[param.human_readable_name], param.default)

        assert(result.exit_code == 0), "CLI Command did not complete successfully"
        assert(os.path.isfile(data_path)), "Config file does not exist"
        os.remove(data_path)

    def test_convert_raw_to_avi(self):


        data_path = 'data/convert_test_depth.dat'
        outfile = data_path.replace('.dat', '.avi')

        write_fake_movie(data_path)

        assert (os.path.isfile(data_path)), "temp depth file not created"

        runner = CliRunner()
        result = runner.invoke(convert_raw_to_avi, [
                                            data_path, '-o', outfile,
                                            '-b', 1000, '--delete'])

        assert (result.exit_code == 0), "CLI command did not complete successfully"
        assert (os.path.isfile(outfile)), "avi file not created"
        assert (not os.path.exists(data_path)), "raw file was not deleted"

        write_fake_movie(data_path)

        assert (os.path.isfile(data_path)), "temp depth file not created"

        result = runner.invoke(convert_raw_to_avi, [
            data_path, '-o', outfile, '-b', 1000,
        ])

        assert (result.exit_code == 0), "CLI command did not complete successfully"
        assert (os.path.isfile(outfile)), "avi file not created"
        os.remove(data_path)
        os.remove(outfile)


    def test_copy_slice(self):

        data_path = 'data/copy_slice_test_depth.dat'

        outfile = data_path.replace('.dat', '.avi')

        write_fake_movie(data_path)

        runner = CliRunner()
        result = runner.invoke(copy_slice, [data_path, '-o', outfile,
                                            '-b', 1000, '--delete'])

        assert (os.path.isfile(outfile)), "slice was not copied correctly"
        assert (not os.path.isfile(data_path)), "input data was not deleted"
        assert (result.exit_code == 0), "CLI command did not complete successfully"
        os.remove(outfile)