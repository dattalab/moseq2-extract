import os
import numpy as np
import numpy.testing as npt
from unittest import TestCase
from tempfile import TemporaryDirectory, NamedTemporaryFile
from moseq2_extract.io.image import write_image, read_image
from skimage.external import tifffile

class TestImageIO(TestCase):
    def test_write_image(self):
        with TemporaryDirectory() as tmp:
            data_path = NamedTemporaryFile(prefix=tmp, suffix=".tiff")

            # make some random ints, don't exceed 16 bit limits
            rnd_img = np.random.randint(low=0, high=100, size=(50, 50)).astype('uint16')
            write_image(data_path.name, rnd_img, scale=False)

            with tifffile.TiffFile(data_path.name) as tif:
                tmp = tif

            image = tmp.asarray().astype('uint16')
            npt.assert_almost_equal(rnd_img, image, 3)
            assert (os.path.exists(data_path.name))


    def test_read_image(self):

        with TemporaryDirectory() as tmp:
            data_path = NamedTemporaryFile(prefix=tmp, suffix=".tiff")

            rnd_img = np.random.randint(low=0, high=100, size=(50, 50)).astype('uint16')

            write_image(data_path.name, rnd_img, scale=True)
            image = read_image(data_path.name, scale=True)

            npt.assert_almost_equal(rnd_img, image, 3)

            write_image(data_path.name, rnd_img, scale=True, scale_factor=(0, 100))
            image = read_image(data_path.name, scale=True)

            npt.assert_almost_equal(rnd_img, image, 3)

            write_image(data_path.name, rnd_img, scale=False)
            image = read_image(data_path.name, scale=False)

            npt.assert_almost_equal(rnd_img, image, 3)
