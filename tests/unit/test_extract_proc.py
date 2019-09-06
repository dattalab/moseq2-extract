import numpy as np
import numpy.testing as npt
import os
import glob
import re
import pytest
import cv2
import joblib
import scipy
from moseq2_extract.io.image import read_image
from moseq2_extract.extract.proc import get_roi, crop_and_rotate_frames,\
    get_frame_features, compute_scalars, clean_frames, get_largest_cc, get_bbox
from moseq2_extract.extract.track import em_get_ll

# https://stackoverflow.com/questions/34504757/
# get-pytest-to-look-within-the-base-directory-of-the-testing-script
@pytest.fixture(scope="function")
def script_loc(request):
    return request.fspath.join('..')


# TODO: Perfect input data to get accurate failure rate
def test_get_flips():
    # original params: frames (crop_rotated frames in usage), flip_file = None, smoothing = None (defaulted at 51)

    # create mock frames of correct shape
    fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))

    tmp_image = np.zeros((80, 80), dtype='int8')
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2

    tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
    center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = fake_mouse

    test_data = np.tile(tmp_image, (100, 1, 1))

    # get all 3 truth flip_file_classifiers
    flip_files = ['tests/test_flip_classifiers/'+f for f in os.listdir('tests/test_flip_classifiers/')]

    # test each classifier separately
    for flip_file in flip_files:
        try:
            clf = joblib.load(flip_file)
        except IOError:
            pytest.fail(f"IOERROR {flip_file}")
            print(f"Could not open file {flip_file}")
            raise

        # run clf on mock data to get mock_probas
        flip_class = np.where(clf.classes_ == 1)[0]

        try:
            mock_probas = clf.predict_proba(
                test_data.reshape((-1, test_data.shape[1] * test_data.shape[2])))
        except:
            pytest.fail(f"model issue: {flip_file}\n")
        # test smoothing option with mock
        for i in range(mock_probas.shape[1]):
            mock_probas[:, i] = scipy.signal.medfilt(mock_probas[:, i], 51)

        # compare result of "mock_probas.argmax(axis=1) == flip_class" with ground truth
        mock_flips = mock_probas.argmax(axis=1) == flip_class
        assert mock_flips.all() == True


    print('done')


# TODO: find correct way to validate ground truth with assertion
def test_get_bground_im():
    # original params: frames (3d numpy array: frames x r x c)

    # create mock frames of correct shape
    fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))

    tmp_image = np.zeros((80, 80), dtype='int8')
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2

    tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
    center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = fake_mouse

    test_data = np.tile(tmp_image, (100, 1, 1))

    # run np.median(frames, 0) and compare result with ground truth
    bground = np.median(test_data, 0)
    print(bground)

    # according to the return statement where shape of bg image must be r x c
    assert bground.shape == (80, 80)

#TODO: Complete io.video unit tests to ensure this function is secure
def test_get_bground_im_file():
    # original params: frames_file, frame_stride=500, med_scale=5, **kwargs

    # create mock existing path for frame_file with necessary info for io.video

    '''
    NOTE: get_bground_im_file contains the following external functions from io.video:
    get_raw_info() [UNIT TEST COMPLETE]
    get_video_info() [UNIT TEST NOT COMPLETE]
    read_frames_raw() [UNIT TEST NOT COMPLETE]
    read_frames() [UNIT TEST NOT COMPLETE]
    '''

    # Once tested above, check frame_idx/store sizes and compare with ground truth

    print('done')

# TODO: Ensure these assertions are sufficient.
def test_get_bbox():
    # original params: roi

    # load in a bunch of ROIs where we have some ground truth
    cwd = str(script_loc)
    bground_list = glob.glob(os.path.join(cwd, 'test_rois/bground*.tiff'))

    for bground in bground_list:
        tmp = read_image(bground, scale=True)
        y, x = np.where(bground > 0)

        if len(y) == 0 or len(x) == 0:
            pytest.fail("x or y == 0")
        else:
            # Apply same transformations on mock
            assert y.min() < y.max()
            assert x.min() < x.max()
            bbox = np.array([[y.min(), x.min()], [y.max(), x.max()]])

    # Compare with ground truth

    print('done-ish: get_bbox check final assertion for actual bbox')


def test_get_roi(script_loc):
    # load in a bunch of ROIs where we have some ground truth
    cwd = str(script_loc)
    bground_list = glob.glob(os.path.join(cwd, 'test_rois/bground*.tiff'))

    for bground in bground_list:
        tmp = read_image(bground, scale=True)
        print(bground)

        # if bground == os.path.join(cwd, 'test_rois/bground_stfp.tiff'):
        #     roi_weights = (1, .1, 1)
        # else:
        #     roi_weights = (1, .1, 1)
        # print(roi_weights)

        if re.search(r'gradient', bground) is not None:
            roi = get_roi(tmp.astype('float32'), depth_range=(750, 900),
                          iters=5000, noise_tolerance=30, gradient_filter=True)
        else:
            roi = get_roi(tmp.astype('float32'), depth_range=(650, 750),
                          iters=5000, noise_tolerance=30)

        fname = os.path.basename(bground)
        dirname = os.path.dirname(bground)
        roi_file = 'roi{}_01.tiff'.format(re.search(r'\_[a-z|A-Z]*',
                                                    fname).group())

        ground_truth = read_image(os.path.join(dirname, roi_file), scale=True)

        frac_nonoverlap_roi1 = np.empty((2,))
        frac_nonoverlap_roi2 = np.empty((2,))

        frac_nonoverlap_roi1[0] = np.mean(
            np.logical_xor(ground_truth, roi[0][0]))

        roi_file2 = 'roi{}_02.tiff'.format(re.search(r'\_[a-z|A-Z]*',
                                                     fname).group())

        if os.path.exists(os.path.join(dirname, roi_file2)):
            ground_truth = read_image(
                os.path.join(dirname, roi_file2), scale=True)
            frac_nonoverlap_roi2[0] = np.mean(np.logical_xor(ground_truth,
                                                             roi[0][1]))
            frac_nonoverlap_roi2[1] = np.mean(np.logical_xor(ground_truth,
                                                             roi[0][0]))
            frac_nonoverlap_roi1[1] = np.mean(np.logical_xor(ground_truth,
                                                             roi[0][1]))

            print(frac_nonoverlap_roi2)
            assert(np.min(frac_nonoverlap_roi2) < .2)

        print(frac_nonoverlap_roi1)
        assert(np.min(frac_nonoverlap_roi1) < .2)

# TODO: use proper comparison assertion and possibly change input data
def test_apply_roi():
    # original params: frames, roi

    '''
    NOTE: this function contains internal function: get_bbox(roi) [SEMI-COMPLETED TEST]
    '''

    # create mock roi and frames of appropriate shapes
    fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))

    tmp_image = np.zeros((80, 80), dtype='int8')
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2

    tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
    center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = fake_mouse

    test_data = np.tile(tmp_image, (100, 1, 1))

    bbox = get_bbox(fake_mouse)

    test_data = test_data[:, bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1]]

    # compare cropped frames with ground truth
    print(test_data)


    print('done')

def test_im_moment_features():
    # original params: IM (2d numpy array): depth image

    # create mock 2d numpy array image of randoms
    fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))

    tmp_image = np.zeros((80, 80), dtype='int8')
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2

    tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
    center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = fake_mouse

    # run computations as in original function to get centroid and ellipse axis info
    cnts, hierarchy = cv2.findContours(
        tmp_image.astype('uint8'),
        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tmp = np.array([cv2.contourArea(x) for x in cnts])

    mouse_cnt = tmp.argmax()

    tmp = cv2.moments(cnts[mouse_cnt])
    num = 2 * tmp['mu11']
    den = tmp['mu20'] - tmp['mu02']

    common = np.sqrt(4 * np.square(tmp['mu11']) + np.square(den))

    # assert that the dict does not contain nans.
    if tmp['m00'] == 0:
        pytest.fail('NANs')

    print('done-ish: moments check final assertions')


#TODO: complete util.py: test_strided_app() unit test to complete this test.
def test_feature_hampel_filter():
    # original params: features, centroid_hampel_span=None, centroid_hampel_sig=3,
#                           angle_hampel_span=None, angle_hampel_sig=3

    '''

    NOTE: this function contains util.py function strided_app() [INCOMPLETE]

    '''


    # use mock features input with appropriate values

    # after strided_app test is complete, continue performing numpy computations

    # compare current result with ground truth
    pytest.fail('not implemented')
    print('not done')


def test_model_smoother():
    # original params: features (dict), ll=None, clips=(-300, -125)

    fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))

    tmp_image = np.zeros((80, 80), dtype='int8')
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2

    tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
    center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = fake_mouse

    # create mock features, ll and clips
    rotations = np.random.rand(100) * 180 - 90
    features = {}

    features['centroid'] = np.tile(center, (len(rotations), 1))
    features['orientation'] = np.deg2rad(rotations)
    clips = (-300, -125)

    # make mock filterframes, progress bar for em_get_ll() [TESTED]
    pytest.fail('not implemented')

    print('test')


def test_crop_and_rotate():

    fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))

    tmp_image = np.zeros((80, 80), dtype='int8')
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2

    tmp_image[center[0]-mouse_dims[0]:center[0]+mouse_dims[0],
              center[1]-mouse_dims[1]:center[1]+mouse_dims[1]] = fake_mouse

    rotations = np.random.rand(100)*180-90
    features = {}

    features['centroid'] = np.tile(center, (len(rotations), 1))
    features['orientation'] = np.deg2rad(rotations)

    fake_movie = np.zeros((len(rotations), tmp_image.shape[0], tmp_image.shape[1]), dtype='float32')

    for i, rotation in enumerate(rotations):

        rot_mat = cv2.getRotationMatrix2D(tuple(center), rotation, 1)
        fake_movie[i] = cv2.warpAffine(tmp_image.astype('float32'), rot_mat, (80, 80))

    cropped_and_rotated = (crop_and_rotate_frames(fake_movie, features=features) > .4).astype(tmp_image.dtype)
    percent_pixels_diff = np.mean(np.abs(cropped_and_rotated-tmp_image)) * 1e2

    assert(percent_pixels_diff < .1)


def test_get_frame_features():

    fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))

    tmp_image = np.zeros((80, 80), dtype='int8')
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2

    tmp_image[center[0]-mouse_dims[0]:center[0]+mouse_dims[0],
              center[1]-mouse_dims[1]:center[1]+mouse_dims[1]] = fake_mouse

    fake_movie = np.tile(tmp_image, (100, 1, 1))
    fake_features, mask = get_frame_features(fake_movie, frame_threshold=0.01)

    npt.assert_almost_equal(fake_features['orientation'], 0, 2)
    npt.assert_almost_equal(fake_features['centroid'],
                            np.tile(center, (fake_movie.shape[0], 1)), .1)
    npt.assert_array_almost_equal(fake_features['axis_length'],
                                  np.tile([30, 20], (fake_movie.shape[0], 1)), .1)


def test_compute_scalars():

    fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))

    tmp_image = np.zeros((80, 80), dtype='int8')
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2

    tmp_image[center[0]-mouse_dims[0]:center[0]+mouse_dims[0],
              center[1]-mouse_dims[1]:center[1]+mouse_dims[1]] = fake_mouse

    fake_movie = np.tile(tmp_image, (100, 1, 1))
    fake_features, mask = get_frame_features(fake_movie, frame_threshold=0.01)
    fake_scalars = compute_scalars(fake_movie, fake_features, min_height=0.01)

    npt.assert_almost_equal(fake_scalars['centroid_x_px'], center[1], .1)
    npt.assert_almost_equal(fake_scalars['centroid_y_px'], center[0], .1)
    npt.assert_almost_equal(fake_scalars['angle'], 0, 2)
    npt.assert_almost_equal(fake_scalars['width_px'], mouse_dims[0]*2, .1)
    npt.assert_almost_equal(fake_scalars['length_px'], mouse_dims[1]*2, .1)
    npt.assert_almost_equal(fake_scalars['height_ave_mm'], 1, 1)
    npt.assert_almost_equal(fake_scalars['velocity_2d_px'], 0, 1)
    npt.assert_almost_equal(fake_scalars['velocity_3d_px'], 0, 1)
    npt.assert_almost_equal(fake_scalars['velocity_theta'], 0, 1)
    npt.assert_almost_equal(fake_scalars['area_px'], np.sum(tmp_image), .1)


def test_get_largest_cc():

    fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))

    tmp_image = np.zeros((80, 80), dtype='int8')
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2

    tmp_image[center[0]-mouse_dims[0]:center[0]+mouse_dims[0],
              center[1]-mouse_dims[1]:center[1]+mouse_dims[1]] = fake_mouse

    fake_movie = np.tile(tmp_image, (100, 1, 1))
    largest_cc_movie = get_largest_cc(fake_movie)

    npt.assert_array_almost_equal(fake_movie, largest_cc_movie, 3)


def test_clean_frames():

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

    fake_movie = np.tile(fake_mouse, (100, 1, 1))
    cleaned_fake_movie = clean_frames(fake_movie, prefilter_time=(3,))
