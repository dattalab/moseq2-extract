import os
import numpy as np
import numpy.testing as npt
from unittest import TestCase
from skimage.external import tifffile
from moseq2_extract.io.image import write_image, read_image, read_tiff_files

class TestImageIO(TestCase):
    def test_write_image(self):

        data_path = 'data/temp_w_image.tiff'

        # make some random ints, don't exceed 16 bit limits
        rnd_img = np.random.randint(low=0, high=100, size=(50, 50)).astype('uint16')
        write_image(data_path, rnd_img, scale=False)

        with tifffile.TiffFile(data_path) as tif:
            tmp = tif

        image = tmp.asarray().astype('uint16')
        npt.assert_almost_equal(rnd_img, image, 3)
        assert (os.path.isfile(data_path))
        os.remove(data_path)

    def test_read_image(self):

        data_path = 'data/temp_r_image.tiff'

        rnd_img = np.random.randint(low=0, high=100, size=(50, 50)).astype('uint16')

        write_image(data_path, rnd_img, scale=True)
        image = read_image(data_path, scale=True)

        npt.assert_almost_equal(rnd_img, image, 3)

        write_image(data_path, rnd_img, scale=True, scale_factor=(0, 100))
        image = read_image(data_path, scale=True)

        npt.assert_almost_equal(rnd_img, image, 3)

        write_image(data_path, rnd_img, scale=False)
        image = read_image(data_path, scale=False)

        npt.assert_almost_equal(rnd_img, image, 3)
        os.remove(data_path)

    def test_read_tiff_files(self):
        tiff_dir = 'data/tiffs/'

        images, paths = read_tiff_files(tiff_dir)
        assert len(images) == len(paths) == 10