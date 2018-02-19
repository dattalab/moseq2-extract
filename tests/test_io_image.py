import numpy.testing as npt
import numpy as np
import pytest
import os
from moseq2.io.image import write_image, read_image
from skimage.external import tifffile


@pytest.fixture(scope='function')
def image_file(tmpdir):
    f = tmpdir.mkdir('test_dir')
    return str(f)


def test_write_image(image_file):

    # make some random ints, don't exceed 16 bit limits

    image_file = os.path.join(image_file, 'test2', 'test_image.tiff')
    rnd_img = np.random.randint(
        low=0, high=100, size=(50, 50)).astype('uint16')
    write_image(image_file, rnd_img, scale=False)

    with tifffile.TiffFile(image_file) as tif:
        tmp = tif

    image = tmp.asarray().astype('uint16')
    npt.assert_almost_equal(rnd_img, image, 3)


def test_read_image(image_file):

    image_file = os.path.join(image_file, 'test2', 'test_image.tiff')
    rnd_img = np.random.randint(
        low=0, high=100, size=(50, 50)).astype('uint16')

    write_image(image_file, rnd_img, scale=True)
    image = read_image(image_file, scale=True)

    npt.assert_almost_equal(rnd_img, image, 3)

    write_image(image_file, rnd_img, scale=True, scale_factor=(0, 100))
    image = read_image(image_file, scale=True)

    npt.assert_almost_equal(rnd_img, image, 3)

    write_image(image_file, rnd_img, scale=False)
    image = read_image(image_file, scale=False)

    npt.assert_almost_equal(rnd_img, image, 3)
