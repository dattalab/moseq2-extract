import numpy as np
import tqdm

def plane_fit3(points):
    """Fit a plane to 3 points (min number of points for fitting a plane)
    Args:
        points (2d numpy array): each row is a group of points, columns correspond to x,y,z

    Returns:
        plane (1d numpy array): linear plane fit-->a*x+b*y+c*z+d

    """
    a=points[1,:]-points[0,:]
    b=points[2,:]-points[0,:]
    # cross prod
    normal=np.array([[a[1]*b[2]-a[2]*b[1]],
                    [a[2]*b[0]-a[0]*b[2]],
                    [a[0]*b[1]-a[1]*b[0]]])
    denom=np.sum(np.square(normal))
    if denom<np.spacing(1):
        plane=np.empty((4,))
        plane[:]=np.nan
    else:
        normal /= np.sqrt(denom)
        d=np.dot(-points[0,:],normal)
        plane=np.hstack((normal.flatten(),d))

    return plane


def plane_ransac(depth_image,depth_range=(650,750),iters=1000,noise_tolerance=30,in_ratio=.1,progress_bar=True):
    """Naive RANSAC implementation for plane fitting
    Args:
        depth_image (2d numpy array): hxw, background image to fit plane to
        depth_range (tuple): min and max depth (in mm) to consider pixels for plane fit
        iters (int): number of RANSAC iterations
        noise_tolerance (float): distance from plane to consider a point an inlier
        in_ratio (float): fraction of points required to consider a plane fit good

    Returns:
        best_plane (1d numpy array): plane fit to data
    """
    use_points=np.logical_and(depth_image>depth_range[0],depth_image<depth_range[1])

    xx , yy =np.meshgrid(np.arange(depth_image.shape[1]),np.arange(depth_image.shape[0]))

    coords=np.vstack((xx[use_points].ravel(),yy[use_points].ravel(),depth_image[use_points].ravel()))
    coords=coords.T

    best_dist=np.inf
    best_num=0

    npoints=np.sum(use_points)

    for i in tqdm.tqdm(np.arange(iters),disable=not progress_bar, desc='Finding plane'):

        sel=coords[np.random.choice(coords.shape[0],3,replace=True),:]
        tmp_plane=plane_fit3(sel)

        if np.all(np.isnan(tmp_plane)):
            continue

        dist=np.abs(np.dot(coords,tmp_plane[:3])+tmp_plane[3])
        inliers=dist<noise_tolerance
        ninliers=np.sum(inliers)

        if (ninliers/npoints)>in_ratio and ninliers>best_num and np.mean(dist)<best_dist:

            best_dist=np.mean(dist)
            best_num=ninliers
            #best_plane=tmp_plane

            # use all consensus samples to fit a better model

            all_data=coords[inliers.flatten(),:]
            mu=np.mean(all_data,0)
            u,s,v=np.linalg.svd(all_data-mu)
            v=v.conj().T
            best_plane=np.hstack((v[:,-1],np.dot(-mu,v[:,-1])))

    # fit the plane to our x,y,z coordinates


    coords=np.vstack((xx.ravel(),yy.ravel(),depth_image.ravel())).T
    dist=np.abs(np.dot(coords,best_plane[:3])+best_plane[3])

    return best_plane , dist
