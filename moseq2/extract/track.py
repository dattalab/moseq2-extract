import numpy as np
import scipy.stats
import statsmodels.stats.correlation_tools as stats_tools
import cv2
import tqdm


def em_iter(data, mean, cov, lamd=.1, epsilon=1e-1, max_iter=25):

    prev_likelihood = 0
    ll = 0

    ndatapoints = data.shape[1]
    pxtheta_raw = np.zeros((ndatapoints,), dtype='float64')

    for i in range(max_iter):
        pxtheta_raw = scipy.stats.multivariate_normal.pdf(
            x=data, mean=mean, cov=cov)
        pxtheta_raw /= np.sum(pxtheta_raw)

        mean = np.sum(data.T*pxtheta_raw, axis=1)
        dx = (data-mean).T
        cov = stats_tools.cov_nearest(
            np.dot(dx * pxtheta_raw, dx.T) + lamd*np.eye(3))

        ll = np.sum(np.log(pxtheta_raw+1e-300))
        delta_likelihood = (ll-prev_likelihood)

        if (delta_likelihood >= 0 and
                delta_likelihood < epsilon*abs(prev_likelihood)):
            break

        prev_likelihood = ll

    return mean, cov


def em_tracking(frames, segment=True, ll_threshold=-30, rho_mean=0, rho_cov=0,
                depth_floor=10, progress_bar=True):
    """The dead-simple tracker, use EM update rules to follow a 3D Gaussian
       around the room!
    Args:
        frames (3d numpy array): nframes x r x c
        segment (bool): use only the largest blob for em updates
        ll_threshold (float): threshold on log likelihood for segmentation
        rho_mean (float): smoothing parameter for the mean
        rho_cov (float): smoothing parameter for the covariance
        depth_floor (float): height in mm for separating mouse from floor

    Returns:
        model_paramters (dict): mean and covariance estimates for each frame
    """
    # initialize the mean and covariance

    nframes, r, c = frames.shape

    xx, yy = np.meshgrid(
        np.arange(frames.shape[2]), np.arange(frames.shape[1]))
    coords = np.vstack((xx.ravel(), yy.ravel()))
    xyz = np.vstack((coords, frames[0, ...].ravel()))
    mean = np.mean(xyz[:, xyz[2, :] > depth_floor], axis=1)
    cov = stats_tools.cov_nearest(np.cov(xyz[:, xyz[2, :] > depth_floor]))

    model_parameters = {
        'mean': np.empty((nframes, 3), 'float64'),
        'cov': np.empty((nframes, 3, 3), 'float64')
    }

    for k, v in model_parameters.items():
        model_parameters[k][:] = np.nan

    frames = frames.reshape(frames.shape[0], frames.shape[1]*frames.shape[2])
    pbar = tqdm.tqdm(total=nframes, disable=not progress_bar,
                     desc='Computing EM')
    i = 0
    repeat = False

    while i < nframes:

        xyz = np.vstack((coords, frames[i, ...]))
        pxtheta_im = scipy.stats.multivariate_normal.logpdf(
            xyz.T, mean, cov).reshape((r, c))

        # segment to find pixels with likely mice, only use those for updating

        if segment and not repeat:
            im2, cnts, hierarchy =\
                cv2.findContours((pxtheta_im > ll_threshold).astype('uint8'),
                                 cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            tmp = np.array([cv2.contourArea(x) for x in cnts])
            if tmp.size == 0:
                repeat = True
                continue
            use_cnt = tmp.argmax()
            mask = np.zeros_like(pxtheta_im)
            cv2.drawContours(mask, cnts, use_cnt, (255), cv2.FILLED)
        else:
            mask = np.ones(pxtheta_im.shape, dtype='bool')

        tmp = mask.ravel() > 0
        tmp[xyz[2, :] <= depth_floor] = 0

        mean_update, cov_update = em_iter(xyz[:, tmp.astype('bool')].T,
                                          mean=mean, cov=cov,
                                          epsilon=.25, max_iter=15, lamd=30)

        # exponential smoothers for mean and covariance if
        # you want (easier to do this offline)
        # leave these set to 0 for now

        mean = (1-rho_mean)*mean_update+rho_mean*mean
        cov = (1-rho_cov)*cov_update+rho_cov*cov

        model_parameters['mean'][i, ...] = mean
        model_parameters['cov'][i, ...] = cov

        # TODO: add the walk-back where we use the
        # raw frames in case our update craps out...

        repeat = False
        i += 1
        pbar.update(1)

    pbar.close()

    return model_parameters


def em_get_ll(frames, mean, cov, progress_bar=True):
    """Get the likelihoods associated with model parameters
    """
    xx, yy = np.meshgrid(np.arange(frames.shape[2]),
                         np.arange(frames.shape[1]))
    coords = np.vstack((xx.ravel(), yy.ravel()))

    nframes, r, c = frames.shape

    ll = np.zeros(frames.shape, dtype='float64')

    for i in tqdm.tqdm(range(nframes), disable=not progress_bar,
                       desc='Computing EM likelihoods'):
        xyz = np.vstack((coords, frames[i, ...].ravel()))
        ll[i, ...] = scipy.stats.multivariate_normal.logpdf(
            xyz.T, mean[i, ...], cov[i, ...]).reshape((r, c))

    return ll
