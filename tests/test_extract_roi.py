import pytest
import numpy.testing as npt
import numpy as np
from moseq2.extract.roi import plane_fit3

def test_plane_fit3():
    # let's create two planes from simple equations and ensure
    # that everything looks kosher ya?

    # x,y plus intercept

    def plane_equation1(xy):
        return 2*xy[:,0]+2*xy[:,1]+2

    def plane_equation2(xy):
        return 1*xy[:,0]+50*xy[:,1]+3

    xy=np.random.randint(low=0,high=100,size=(3,2))
    xyz=np.hstack((xy,plane_equation1(xy)[:,np.newaxis])).astype('float64')
    a=plane_fit3(xyz)
    norma=-a/a[2]

    npt.assert_almost_equal(norma[[0,1,3]],np.array([2,2,2]),3)
    npt.assert_almost_equal(np.dot(a[:3],xyz.T-a[3]),[0,0,0],3)

    xyz=np.hstack((xy,plane_equation2(xy)[:,np.newaxis])).astype('float64')
    a=plane_fit3(xyz)
    norma=-a/a[2]

    npt.assert_almost_equal(norma[[0,1,3]],np.array([1,50,3]),3)
    npt.assert_almost_equal(np.dot(a[:3],xyz.T-a[3]),[0,0,0],3)
