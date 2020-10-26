import numpy as np
import numpy.testing as npt
from unittest import TestCase
import statsmodels.stats.correlation_tools as stats_tools
from moseq2_extract.extract.track import em_get_ll, em_tracking

def make_fake_movie():
    edge_size = 40
    points = np.arange(-edge_size, edge_size)
    sig1 = 10
    sig2 = 20
    height = 50

    kernel = np.exp(-(points ** 2.0) / (2.0 * sig1 ** 2.0))
    kernel2 = np.exp(-(points ** 2.0) / (2.0 * sig2 ** 2.0))

    kernel_full = np.outer(kernel, kernel2)
    kernel_full /= np.max(kernel_full)
    kernel_full *= height

    fake_mouse = kernel_full
    fake_mouse[fake_mouse < 5] = 0
    im_size = (424, 512)
    tmp_image = np.zeros(im_size, dtype='int16')
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2
    tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
    center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = fake_mouse

    xx, yy = np.meshgrid(np.arange(tmp_image.shape[1]), np.arange(tmp_image.shape[0]))
    coords = np.vstack((xx.ravel(), yy.ravel(), tmp_image.ravel()))
    tmp_cov = stats_tools.cov_nearest(np.cov(coords[:, coords[2, :] > 10]))

    nframes = 100
    fake_movie = np.tile(tmp_image, (nframes, 1, 1))

    return fake_movie, tmp_cov, center, im_size, nframes


class TestEMTracking(TestCase):

    def test_em_get_ll(self):

        # draw some random samples
        fake_movie, tmp_cov, center, im_size, nframes = make_fake_movie()

        ll = em_get_ll(frames=fake_movie,
                       mean=np.tile(np.array([center[1], center[0], 50]),
                                    (nframes, 1)),
                       cov=np.tile(tmp_cov, (nframes, 1, 1)))

        for ll_im in ll:
            npt.assert_almost_equal(np.unravel_index(np.argmax(ll_im), im_size), center)

    def test_em_tracking(self):

        # draw some random samples
        fake_movie, tmp_cov, center, im_size, nframes = make_fake_movie()

        for init in ['raw', 'min', 'med']:
            parameters = em_tracking(frames=fake_movie, raw_frames=fake_movie, init_method=init)

            # this is very loose atm, need to figure out what's going on here...
            for mu in parameters['mean']:
                npt.assert_allclose(mu[:2], center[::-1], atol=5, rtol=0)