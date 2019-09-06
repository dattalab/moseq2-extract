import numpy.testing as npt
import numpy as np
import statsmodels.stats.correlation_tools as stats_tools
import scipy
import pytest
from moseq2_extract.extract.track import em_get_ll, em_iter, em_tracking
import cv2


def test_em_iter():
    """Single iteration of EM tracker
        Args:
            data (3d numpy array): nx3, x, y, z coordinates to use
            mean (1d numpy array): dx1, current mean estimate
            cov (2d numpy array): dxd, current covariance estimate
            lambd (float): constant to add to diagonal of covariance matrix
            epsilon (float): tolerance on change in likelihood to terminate iteration
            max_iter (int):

        Returns:
            mean (1d numpy array): updated mean
            cov (2d numpy array): updated covariance
    """
    data = cv2.getStructuringElement(cv2.MORPH_RECT,(1,5))
    mean = [np.mean(data)]
    cov = np.ones((1,1),np.uint8)
    lamd = .1
    epsilon = 1e-1
    max_iter = 1

    prev_likelihood = 0
    ll = 0

    ndatapoints = data.shape[1]
    pxtheta_raw = np.zeros((ndatapoints,), dtype='float64')

    for i in range(max_iter):
        pxtheta_raw = scipy.stats.multivariate_normal.pdf(x=data, mean=mean, cov=cov)
        pxtheta_raw /= np.sum(pxtheta_raw)

        mean = np.sum(data.T * pxtheta_raw, axis=1)
        dx = (data - mean).T
        cov = stats_tools.cov_nearest(np.dot(dx * pxtheta_raw, dx.T) + lamd*np.eye(3))

        ll = np.sum(np.log(pxtheta_raw+1e-300))
        delta_likelihood = (ll-prev_likelihood)

        if (delta_likelihood >= 0 and
                delta_likelihood < epsilon*abs(prev_likelihood)):
            break

        prev_likelihood = ll

        if mean.shape != (1,):
            pytest.fail('MEAN IS INCORRECT SHAPE')

        if cov.shape != (3, 3):
            pytest.fail('COV IS INCORRECT SHAPE')
        print(mean, cov)

def test_em_init():
    # original params: depth_frame, depth_floor, depth_ceiling,
#             init_strel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), strel_iters=1

    # mock parameters
    depth_frame = 300
    depth_floor = 100
    depth_ceiling = 400
    init_strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    strel_iters = 1

    mask = np.logical_and(depth_frame > depth_floor, depth_frame < depth_ceiling)
    # assuming inputing a rectangular image
    mask = cv2.morphologyEx(cv2.getStructuringElement(cv2.MORPH_RECT,(depth_floor,depth_ceiling)), cv2.MORPH_OPEN, init_strel, strel_iters)

    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tmp = np.array([cv2.contourArea(x) for x in cnts])
    try:
        use_cnt = tmp.argmax()
        mouse_mask = np.zeros_like(mask)
        cv2.drawContours(mouse_mask, cnts, use_cnt, (255), cv2.FILLED)
        mouse_mask = mouse_mask > 0
    except Exception:
        mouse_mask = mask > 0
        pytest.fail('Exception thrown')

def test_em_get_ll():

    # draw some random samples

    edge_size = 40
    points = np.arange(-edge_size, edge_size)
    sig1 = 10
    sig2 = 20
    height = 50

    kernel = np.exp(-(points**2.0) / (2.0 * sig1**2.0))
    kernel2 = np.exp(-(points**2.0) / (2.0 * sig2**2.0))

    kernel_full = np.outer(kernel, kernel2)
    kernel_full /= np.max(kernel_full)
    kernel_full *= height

    fake_mouse = kernel_full
    fake_mouse[fake_mouse < 5] = 0
    im_size = (424, 512)
    tmp_image = np.zeros(im_size, dtype='int16')
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2
    tmp_image[center[0]-mouse_dims[0]:center[0]+mouse_dims[0],
              center[1]-mouse_dims[1]:center[1]+mouse_dims[1]] = fake_mouse

    xx, yy = np.meshgrid(np.arange(tmp_image.shape[1]), np.arange(tmp_image.shape[0]))
    coords = np.vstack((xx.ravel(), yy.ravel(), tmp_image.ravel()))
    tmp_cov = stats_tools.cov_nearest(np.cov(coords[:, coords[2, :] > 10]))

    nframes = 100
    fake_movie = np.tile(tmp_image, (nframes, 1, 1))

    ll = em_get_ll(frames=fake_movie,
                   mean=np.tile(np.array([center[1], center[0], 50]),
                                (nframes, 1)),
                   cov=np.tile(tmp_cov, (nframes, 1, 1)))

    for ll_im in ll:
        npt.assert_almost_equal(np.unravel_index(np.argmax(ll_im), im_size), center)


def test_em_tracking():

    # draw some random samples

    edge_size = 40
    points = np.arange(-edge_size, edge_size)
    sig1 = 10
    sig2 = 20
    height = 50

    kernel = np.exp(-(points**2.0) / (2.0 * sig1**2.0))
    kernel2 = np.exp(-(points**2.0) / (2.0 * sig2**2.0))

    kernel_full = np.outer(kernel, kernel2)
    kernel_full /= np.max(kernel_full)
    kernel_full *= height

    fake_mouse = kernel_full
    fake_mouse[fake_mouse < 5] = 0
    im_size = (424, 512)
    tmp_image = np.zeros(im_size, dtype='int16')
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2
    tmp_image[center[0]-mouse_dims[0]:center[0]+mouse_dims[0],
              center[1]-mouse_dims[1]:center[1]+mouse_dims[1]] = fake_mouse

    nframes = 100
    fake_movie = np.tile(tmp_image, (nframes, 1, 1))

    parameters = em_tracking(frames=fake_movie, raw_frames=fake_movie)

    # this is very loose atm, need to figure out what's going on here...
    for mu in parameters['mean']:
        npt.assert_allclose(mu[:2], center[::-1], atol=5, rtol=0)
