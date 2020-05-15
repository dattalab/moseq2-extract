import os
import cv2
import click
import shutil
import numpy as np
from pathlib import Path
import ruamel.yaml as yaml
import numpy.testing as npt
from unittest import TestCase
from click.testing import CliRunner
from tempfile import TemporaryDirectory, NamedTemporaryFile
from moseq2_extract.cli import find_roi, extract, download_flip_file, generate_config, convert_raw_to_avi, copy_slice

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

    def test_extract(self):

        with TemporaryDirectory() as tmp:
            data_path = Path(NamedTemporaryFile(prefix=tmp+'/', suffix=".dat").name)

            write_fake_movie(data_path)
            assert data_path.is_file(), "fake movie was not written"

            runner = CliRunner()
            result = runner.invoke(extract, [str(data_path),
                                             '--output-dir', str(data_path.parent),
                                             #'--angle-hampel-span', 5, # add auto fix for incorrect param inputs
                                             #'--centroid-hampel-span', 5
                                             ],
                                   catch_exceptions=False)

            assert(result.exit_code == 0), "CLI command did not successfully complete"
            assert ('done.txt' in os.listdir(os.path.dirname(data_path))), "extraction was interrupted"

    def test_find_roi(self):

        with TemporaryDirectory() as tmp:
            data_path = NamedTemporaryFile(prefix=tmp+'/', suffix=".dat")

            write_fake_movie(data_path.name)

            runner = CliRunner()
            result = runner.invoke(find_roi, [data_path.name, '--output-dir', tmp+'/out/'])

            assert(result.exit_code == 0), "CLI command did not successfully complete"
            assert (len(list(Path(tmp+'/out/').glob("*.tiff"))) == 3), \
                "ROI files were not generated in the correct directory"

    def test_download_flip_file(self):

        with TemporaryDirectory() as tmp:
            data_path = NamedTemporaryFile(prefix=tmp+'/', suffix=".yaml")

            runner = CliRunner()
            result = runner.invoke(download_flip_file, [data_path.name, '--output-dir', tmp], input='0\n')
            assert (result.exit_code == 0), "CLI command did not complete successfully"
            assert (len(list(Path(tmp).glob("*.pkl"))) > 0), "Flip file was not downloaded correctly"


    def test_generate_config(self):

        with TemporaryDirectory() as tmp:
            data_path = NamedTemporaryFile(prefix=tmp+'/', suffix=".yaml")

            runner = CliRunner()
            result = runner.invoke(generate_config, ['--output-file', data_path.name])
            yaml_data = yaml.load(tmp, Loader=yaml.RoundTripLoader)
            temp_p = extract.params
            params = [param for param in temp_p if type(temp_p) is click.core.Option]

            for param in params:
                npt.assert_equal(yaml_data[param.human_readable_name], param.default)

            assert(result.exit_code == 0), "CLI Command did not complete successfully"
            assert(Path(data_path.name).is_file()), "Config file does not exist"

    def test_convert_raw_to_avi(self):

        with TemporaryDirectory() as tmp:
            data_path = Path(NamedTemporaryFile(prefix=tmp+'/', suffix=".dat").name)

            outfile = data_path.parent.joinpath(data_path.name.replace('dat', 'avi'))


            write_fake_movie(data_path)

            runner = CliRunner()
            result = runner.invoke(convert_raw_to_avi, [
                                                str(data_path), '-o', str(outfile),
                                                '-b', 1000, '--delete'
                                                        ])

            assert (outfile.is_file()), "avi file not created"
            assert (not data_path.is_file()), "raw file was not deleted"
            assert (result.exit_code == 0), "CLI command did not complete successfully"

            write_fake_movie(data_path)

            result = runner.invoke(convert_raw_to_avi, [
                str(data_path), '-o', str(outfile), '-b', 1000,
            ])

            assert (outfile.is_file()), "avi file not created"
            assert (result.exit_code == 0), "CLI command did not complete successfully"

    def test_copy_slice(self):

        with TemporaryDirectory() as tmp:
            data_path = Path(NamedTemporaryFile(prefix=tmp+'/', suffix=".dat").name)

            outfile = data_path.parent.joinpath(data_path.name.replace('.dat', '.avi'))

            write_fake_movie(data_path)

            runner = CliRunner()
            result = runner.invoke(copy_slice, [str(data_path), '-o', str(outfile),
                                                '-b', 1000, '--delete'])

            assert (outfile.is_file()), "slice was not copied correctly"
            assert (not data_path.is_file()), "input data was not deleted"
            assert (result.exit_code == 0), "CLI command did not complete successfully"