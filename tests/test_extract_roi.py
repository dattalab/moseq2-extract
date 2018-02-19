import numpy.testing as npt
import numpy as np
from moseq2.extract.roi import plane_fit3, plane_ransac


def test_plane_fit3():
    # let's create two planes from simple equations and ensure
    # that everything looks kosher ya?

    # x,y plus intercept

    def plane_equation1(xy):
        return 2*xy[:, 0]+2*xy[:, 1]+2

    def plane_equation2(xy):
        return 1*xy[:, 0]+50*xy[:, 1]+3

    xy = np.random.randint(low=0, high=100, size=(3, 2))
    xyz = np.hstack((xy, plane_equation1(xy)[:, np.newaxis])).astype('float64')
    a = plane_fit3(xyz)
    norma = -a/a[2]

    npt.assert_almost_equal(norma[[0, 1, 3]], np.array([2, 2, 2]), 3)
    # npt.assert_almost_equal(np.dot(a[:3],xyz.T-a[3]),[0,0,0],3)

    xyz = np.hstack((xy, plane_equation2(xy)[:, np.newaxis])).astype('float64')
    a = plane_fit3(xyz)
    norma = -a/a[2]

    npt.assert_almost_equal(norma[[0, 1, 3]], np.array([1, 50, 3]), 3)
    # npt.assert_almost_equal(np.dot(a[:3],xyz.T-a[3]),[0,0,0],3)


def test_plane_ransac():

    # make a plane where we can test adding noise...

    def plane_equation_noisy(xy, noise_scale=0):
        return (2*xy[:, 0]+5*xy[:, 1]+1 +
                np.random.normal(0, noise_scale, size=(xy.shape[0],)))

    xx, yy = np.meshgrid(np.arange(0, 50), np.arange(0, 50))

    xy = np.vstack((xx.ravel(), yy.ravel()))

    # low noise regime

    z = plane_equation_noisy(xy.T, noise_scale=1)
    tmp_img = z.reshape(xx.shape)

    a = plane_ransac(tmp_img, depth_range=(0, 1000),
                     iters=1000, noise_tolerance=10)
    norma = -a[0]/a[0][2]

    npt.assert_almost_equal(norma[[0, 1]], np.array([2, 5]), 1)

    # high noise regime

    z = plane_equation_noisy(xy.T, noise_scale=10)
    tmp_img = z.reshape(xx.shape)

    a = plane_ransac(tmp_img, depth_range=(0, 1000),
                     iters=10000, noise_tolerance=10)
    norma = -a[0]/a[0][2]

    npt.assert_almost_equal(norma[[0, 1]], np.array([2, 5]), 1)
