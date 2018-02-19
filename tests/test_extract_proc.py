import numpy as np
import os
import glob
import re
import pytest
from moseq2.io.image import read_image
from moseq2.extract.proc import get_roi


# https://stackoverflow.com/questions/34504757/
# get-pytest-to-look-within-the-base-directory-of-the-testing-script
@pytest.fixture(scope="module")
def script_loc(request):
    return request.fspath.join('..')


def test_get_roi(script_loc):
    # load in a bunch of ROIs where we have some ground truth
    cwd = str(script_loc)
    bground_list = glob.glob(os.path.join(cwd, 'test_rois/bground*.tiff'))

    for bground in bground_list:
        tmp = read_image(bground, scale=True)
        roi = get_roi(tmp.astype('float32'), depth_range=(650, 750),
                      iters=1000, noise_tolerance=10)

        fname = os.path.basename(bground)
        dirname = os.path.dirname(bground)
        roi_file = 'roi{}_01.tiff'.format(re.search(r'\_[a-z|A-Z]*',
                                                    fname).group())

        ground_truth = read_image(os.path.join(dirname, roi_file), scale=True)
        frac_nonoverlap = np.mean(np.logical_xor(ground_truth, roi[0][0]))

        assert(frac_nonoverlap < .1)

        roi_file2 = 'roi{}_02.tiff'.format(re.search(r'\_[a-z|A-Z]*',
                                                     fname).group())

        if os.path.exists(os.path.join(dirname, roi_file2)):
            ground_truth = read_image(
                os.path.join(dirname, roi_file2), scale=True)
            frac_nonoverlap = np.mean(np.logical_xor(ground_truth, roi[0][1]))

            assert(frac_nonoverlap < .1)
