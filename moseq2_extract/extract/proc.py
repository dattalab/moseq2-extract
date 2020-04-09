from moseq2_extract.util import convert_pxs_to_mm, strided_app
import moseq2_extract.io.video
import moseq2_extract.extract.roi
import numpy as np
import skimage.measure
import skimage.morphology
import scipy.stats
import scipy.signal
import scipy.interpolate
import cv2
from tqdm.auto import tqdm
import joblib


def get_flips(frames, flip_file=None, smoothing=None):
    '''
    Predicts frames where mouse orientation is flipped to later correct.
    Parameters
    ----------
    frames (3d numpy array): frames x r x c, cropped mouse
    flip_file (str): path to joblib dump of scipy random forest classifier
    smoothing (int): kernel size for median filter smoothing of random forest probabilities

    Returns
    -------
    flips (bool array):  true for flips
    '''

    try:
        clf = joblib.load(flip_file)
    except IOError:
        print(f"Could not open file {flip_file}")
        raise

    flip_class = np.where(clf.classes_ == 1)[0]
    probas = clf.predict_proba(
        frames.reshape((-1, frames.shape[1] * frames.shape[2])))

    if smoothing:
        for i in range(probas.shape[1]):
            probas[:, i] = scipy.signal.medfilt(probas[:, i], smoothing)

    flips = probas.argmax(axis=1) == flip_class

    return flips


def get_largest_cc(frames, progress_bar=False):
    '''
    Returns largest connected component blob in image
    Parameters
    ----------
    frames (3d numpy array): frames x r x c, uncropped mouse
    progress_bar (bool): display progress bar

    Returns
    -------
    flips (3d bool array):  frames x r x c, true where blob was found
    '''

    foreground_obj = np.zeros((frames.shape), 'bool')

    for i in tqdm(range(frames.shape[0]), disable=not progress_bar, desc='CC'):
        nb_components, output, stats, centroids =\
            cv2.connectedComponentsWithStats(frames[i, ...], connectivity=4)
        szs = stats[:, -1]
        foreground_obj[i, ...] = output == szs[1:].argmax()+1

    return foreground_obj


def get_bground_im(frames):
    '''
    Returns background
    Parameters
    ----------
    frames (3d numpy array): frames x r x c, uncropped mouse

    Returns
    -------
    bground (2d numpy array):  r x c, background image
    '''

    bground = np.median(frames, 0)
    return bground


def get_bground_im_file(frames_file, frame_stride=500, med_scale=5, **kwargs):
    '''
    Returns background from file
    Parameters
    ----------
    frames_file (str): path to data with frames
    frame_stride (int): stride size between frames for median bground calculation
    med_scale (int): kernel size for median blur for background images.
    kwargs

    Returns
    -------
    bground (2d numpy array):  r x c, background image
    '''

    try:
        if frames_file.endswith('dat'):
            finfo = moseq2_extract.io.video.get_raw_info(frames_file)
        elif frames_file.endswith('avi'):
            finfo = moseq2_extract.io.video.get_video_info(frames_file)
        elif frames_file.endswith('mkv'):
            finfo = moseq2_extract.io.video.get_raw_info(frames_file)
    except AttributeError as e:
        finfo = moseq2_extract.io.video.get_raw_info(frames_file)

    frame_idx = np.arange(0, finfo['nframes'], frame_stride)
    frame_store = np.zeros((len(frame_idx), finfo['dims'][1], finfo['dims'][0]))

    for i, frame in enumerate(frame_idx):
        try:
            if frames_file.endswith('dat'):
                frs = moseq2_extract.io.video.read_frames_raw(frames_file, int(frame)).squeeze()
            elif frames_file.endswith('avi'):
                frs = moseq2_extract.io.video.read_frames(frames_file, [int(frame)]).squeeze()
            elif frames_file.endswith('mkv'):
                frs = moseq2_extract.io.video.read_frames(frames_file, [int(frame)]).squeeze()
        except AttributeError as e:
            frs = moseq2_extract.io.video.read_frames_raw(frames_file, int(frame), **kwargs).squeeze()

        frame_store[i, ...] = cv2.medianBlur(frs, med_scale)

    return get_bground_im(frame_store)


def get_bbox(roi):
    '''
    Given a binary mask, return an array with the x and y boundaries
    Parameters
    ----------
    roi (2d np.ndarray): ROI boolean mask to calculate bounding box.

    Returns
    -------
    bbox (2d np.ndarray): Bounding Box around ROI
    '''

    y, x = np.where(roi > 0)

    if len(y) == 0 or len(x) == 0:
        return None
    else:
        bbox = np.array([[y.min(), x.min()], [y.max(), x.max()]])
        return bbox


def get_roi(depth_image,
            strel_dilate=cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
            dilate_iters=1,
            strel_erode=None,
            noise_tolerance=30,
            weights=(1, .1, 1),
            overlap_roi=None,
            gradient_filter=False,
            gradient_kernel=7,
            gradient_threshold=3000,
            fill_holes=True,
            gui=False,
            verbose=0,
            **kwargs):
    '''
    Get an ROI using RANSAC plane fitting and simple blob features
    Parameters
    ----------
    depth_image (2d np.ndarray): Singular depth image frame.
    strel_dilate (cv2.StructuringElement - Rectangle): dilation shape to use.
    dilate_iters (int): number of dilation iterations.
    strel_erode (int): image erosion kernel size.
    noise_tolerance (int): threshold to use for noise filtering.
    weights (tuple): weights describing threshold to accept ROI.
    overlap_roi (np.ndarray): list of ROI boolean arrays to possibly combine.
    gradient_filter (bool): Boolean for whether to use a gradient filter.
    gradient_kernel (tuple): Kernel size of length 2, e.g. (1, 1.5)
    gradient_threshold (int): Threshold for noise gradient filtering
    fill_holes (bool): Boolean to fill any missing regions within the ROI.
    gui (bool): Boolean for whether function is running on GUI.
    verbose (bool): Boolean for whether to display progress
    kwargs

    Returns
    -------
    rois (list): list of 2d roi images.
    roi_plane (2d np.ndarray): computed ROI Plane using RANSAC.
    bboxes (list): list of computed bounding boxes for each respective ROI.
    label_im (list): list of scikit-image image properties
    ranks (list): list of ROI ranks.
    shape_index (list): list of rank means.
    '''

    if gradient_filter:
        gradient_x = np.abs(cv2.Sobel(depth_image, cv2.CV_64F,
                                      1, 0, ksize=gradient_kernel))
        gradient_y = np.abs(cv2.Sobel(depth_image, cv2.CV_64F,
                                      0, 1, ksize=gradient_kernel))
        mask = np.logical_and(gradient_x < gradient_threshold, gradient_y < gradient_threshold)
    else:
        mask = None

    roi_plane, dists = moseq2_extract.extract.roi.plane_ransac(
        depth_image, noise_tolerance=noise_tolerance, mask=mask, gui=gui, verbose=verbose, **kwargs)
    dist_ims = dists.reshape(depth_image.shape)

    if gradient_filter:
        dist_ims[~mask] = np.inf

    bin_im = dist_ims < noise_tolerance

    # anything < noise_tolerance from the plane is part of it

    label_im = skimage.measure.label(bin_im)
    region_properties = skimage.measure.regionprops(label_im)

    areas = np.zeros((len(region_properties),))
    extents = np.zeros_like(areas)
    dists = np.zeros_like(extents)

    # get the max distance from the center, area and extent

    center = np.array(depth_image.shape)/2

    for i, props in enumerate(region_properties):
        areas[i] = props.area
        extents[i] = props.extent
        tmp_dists = np.sqrt(np.sum(np.square(props.coords-center), 1))
        dists[i] = tmp_dists.max()

    # rank features

    ranks = np.vstack((scipy.stats.rankdata(-areas, method='max'),
                       scipy.stats.rankdata(-extents, method='max'),
                       scipy.stats.rankdata(dists, method='max')))
    weight_array = np.array(weights, 'float32')
    shape_index = np.mean(np.multiply(ranks.astype('float32'), weight_array[:, np.newaxis]), 0).argsort()

    # expansion microscopy on the roi

    rois = []
    bboxes = []

    for shape in shape_index:
        roi = np.zeros_like(depth_image)
        roi[region_properties[shape].coords[:, 0],
            region_properties[shape].coords[:, 1]] = 1
        if strel_dilate is not None:
            roi = cv2.dilate(roi, strel_dilate, iterations=dilate_iters)
        if strel_erode is not None:
            roi = cv2.erode(roi, strel_erode, iterations=1)
        if fill_holes:
            roi = scipy.ndimage.morphology.binary_fill_holes(roi)

        # roi=skimage.morphology.dilation(roi,dilate_element)
        rois.append(roi)
        bboxes.append(get_bbox(roi))

    if overlap_roi is not None:
        overlaps = np.zeros_like(areas)

        for i in range(len(rois)):
            overlaps[i] = np.sum(np.logical_and(overlap_roi, rois[i]))

        del_roi = np.argmax(overlaps)
        del rois[del_roi]
        del bboxes[del_roi]

    return rois, roi_plane, bboxes, label_im, ranks, shape_index


def apply_roi(frames, roi):
    '''
    Apply ROI to data, consider adding constraints (e.g. mod32==0).
    Parameters
    ----------
    frames (3d np.ndarray): input frames to apply ROI.
    roi (2d np.ndarray): selected ROI to extract from input images.

    Returns
    -------
    cropped_frames (3d np.ndarray): Frames cropped around ROI Bounding Box.
    '''

    # yeah so fancy indexing slows us down by 3-5x
    cropped_frames = frames*roi
    bbox = get_bbox(roi)
    # cropped_frames[:,roi==0]=0
    cropped_frames = cropped_frames[:, bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1]]
    return cropped_frames


def im_moment_features(IM):
    '''
    Use the method of moments and centralized moments to get image properties.
    Parameters
    ----------
    IM (2d numpy array): depth image

    Returns
    -------
    features (dict): returns a dictionary with orientation,
        centroid, and ellipse axis length
    '''

    tmp = cv2.moments(IM)
    num = 2*tmp['mu11']
    den = tmp['mu20']-tmp['mu02']

    common = np.sqrt(4*np.square(tmp['mu11'])+np.square(den))

    if tmp['m00'] == 0:
        features = {
            'orientation': np.nan,
            'centroid': np.nan,
            'axis_length': [np.nan, np.nan]}
    else:
        features = {
            'orientation': -.5*np.arctan2(num, den),
            'centroid': [tmp['m10']/tmp['m00'], tmp['m01']/tmp['m00']],
            'axis_length': [2*np.sqrt(2)*np.sqrt((tmp['mu20']+tmp['mu02']+common)/tmp['m00']),
                            2*np.sqrt(2)*np.sqrt((tmp['mu20']+tmp['mu02']-common)/tmp['m00'])]
        }

    return features


def clean_frames(frames, prefilter_space=(3,), prefilter_time=None,
                 strel_tail=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                 iters_tail=None, frame_dtype='uint8',
                 strel_min=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                 iters_min=None, progress_bar=True, gui=False, verbose=0):
    '''
    Simple filtering, median filter and morphological opening
    Parameters
    ----------
    frames (3d np.ndarray): Frames (nframes x r x c) to filter.
    prefilter_space (tuple): kernel size for spatial filtering
    prefilter_time (tuple): kernel size for temporal filtering
    strel_tail (cv2.StructuringElement): Element for tail filtering.
    iters_tail (int): number of iterations to run opening
    frame_dtype (str): frame encodings
    strel_min (int): minimum kernel size
    iters_min (int): minimum number of filtering iterations
    progress_bar (bool): display progress bar
    gui (bool): indicate GUI is executing function
    verbose (bool): display progress

    Returns
    -------
    filtered_frames (3d np array): frame x r x c
    '''

    # seeing enormous speed gains w/ opencv
    filtered_frames = frames.copy().astype(frame_dtype)

    if verbose == 0:
        progress_bar = False

    for i in tqdm(range(frames.shape[0]),
                       disable=not progress_bar, desc='Cleaning frames'):

        if iters_min is not None and iters_min > 0:
            filtered_frames[i, ...] = cv2.erode(filtered_frames[i, ...], strel_min, iters_min)

        if prefilter_space is not None and np.all(np.array(prefilter_space) > 0):
            for j in range(len(prefilter_space)):
                filtered_frames[i, ...] = cv2.medianBlur(filtered_frames[i, ...], prefilter_space[j])

        if iters_tail is not None and iters_tail > 0:
            filtered_frames[i, ...] = cv2.morphologyEx(
                filtered_frames[i, ...], cv2.MORPH_OPEN, strel_tail, iters_tail)


    if prefilter_time is not None and np.all(np.array(prefilter_time) > 0):
        for j in range(len(prefilter_time)):
            filtered_frames = scipy.signal.medfilt(
                filtered_frames, [prefilter_time[j], 1, 1])

    return filtered_frames


def get_frame_features(frames, frame_threshold=10, mask=np.array([]),
                       mask_threshold=-30, use_cc=False, progress_bar=True, gui=False, verbose=0):
    '''
    Use image moments to compute features of the largest object in the frame
    Parameters
    ----------
    frames (3d np.ndarray): input frames
    frame_threshold (int): threshold in mm separating floor from mouse
    mask (3d np.ndarray): input frame mask for parts not to filter.
    mask_threshold (int): threshold to include regions into mask.
    use_cc (bool): Use connected components.
    progress_bar (bool): Display progress bar.
    gui (bool): indicate GUI is executing function
    verbose (bool): display progress

    Returns
    -------
    features (dict of lists): dictionary with simple image features
    mask (3d np.ndarray): input frame mask.
    '''

    features = []
    nframes = frames.shape[0]

    if type(mask) is np.ndarray and mask.size > 0:
        has_mask = True
    else:
        has_mask = False
        mask = np.zeros((frames.shape), 'uint8')

    features = {
        'centroid': np.empty((nframes, 2)),
        'orientation': np.empty((nframes,)),
        'axis_length': np.empty((nframes, 2))
    }

    for k, v in features.items():
        features[k][:] = np.nan

    if verbose == 0:
            progress_bar = False
    for i in tqdm(range(nframes), disable=not progress_bar, desc='Computing moments'):

        frame_mask = frames[i, ...] > frame_threshold

        if use_cc:
            cc_mask = get_largest_cc((frames[[i], ...] > mask_threshold).astype('uint8')).squeeze()
            frame_mask = np.logical_and(cc_mask, frame_mask)

        if has_mask:
            frame_mask = np.logical_and(frame_mask, mask[i, ...] > mask_threshold)
        else:
            mask[i, ...] = frame_mask

        cnts, hierarchy = cv2.findContours(
            frame_mask.astype('uint8'),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tmp = np.array([cv2.contourArea(x) for x in cnts])

        if tmp.size == 0:
            continue

        mouse_cnt = tmp.argmax()

        for key, value in im_moment_features(cnts[mouse_cnt]).items():
            features[key][i] = value

    return features, mask


def crop_and_rotate_frames(frames, features, crop_size=(80, 80),
                           progress_bar=True, gui=False, verbose=0):
    '''
    Crops mouse from image and orients it s.t it is always facing east.
    Parameters
    ----------
    frames (3d np.ndarray): frames to crop and rotate
    features (dict): dict of extracted features, found in result_00.h5 files.
    crop_size (tuple): size of cropped image.
    progress_bar (bool): Display progress bar.
    gui (bool): indicate GUI is executing function
    verbose (bool): display progress

    Returns
    -------
    cropped_frames (3d np.ndarray): Crop and rotated frames.
    '''

    nframes = frames.shape[0]
    cropped_frames = np.zeros((nframes, crop_size[0], crop_size[1]), frames.dtype)
    win = (crop_size[0] // 2, crop_size[1] // 2 + 1)
    border = (crop_size[1], crop_size[1], crop_size[0], crop_size[0])
    if verbose == 0:
        progress_bar = False
    for i in tqdm(range(frames.shape[0]), disable=not progress_bar, desc='Rotating'):

        if np.any(np.isnan(features['centroid'][i, :])):
            continue

        # use_frame = np.pad(frames[i, ...], (crop_size, crop_size), 'constant', constant_values=0)
        use_frame = cv2.copyMakeBorder(frames[i, ...], *border, cv2.BORDER_CONSTANT, 0)

        rr = np.arange(features['centroid'][i, 1]-win[0],
                       features['centroid'][i, 1]+win[1]).astype('int16')
        cc = np.arange(features['centroid'][i, 0]-win[0],
                       features['centroid'][i, 0]+win[1]).astype('int16')

        rr = rr+crop_size[0]
        cc = cc+crop_size[1]

        if (np.any(rr >= use_frame.shape[0]) or np.any(rr < 1)
                or np.any(cc >= use_frame.shape[1]) or np.any(cc < 1)):
            continue

        rot_mat = cv2.getRotationMatrix2D((crop_size[0] // 2, crop_size[1] // 2),
                                          -np.rad2deg(features['orientation'][i]), 1)
        cropped_frames[i, :, :] = cv2.warpAffine(use_frame[rr[0]:rr[-1], cc[0]:cc[-1]],
                                                 rot_mat, (crop_size[0], crop_size[1]))

    return cropped_frames


def compute_scalars(frames, track_features, min_height=10, max_height=100, true_depth=673.1):
    '''
    Computes scalars
    Parameters
    ----------
    frames (3d np.ndarray): frames x r x c, uncropped mouse
    track_features (dict):  dictionary with tracking variables (centroid and orientation)
    min_height (float): minimum height of the mouse
    max_height (float): maximum height of the mouse
    true_depth (float): detected true depth

    Returns
    -------
    features (dict): dictionary of scalars
    '''

    nframes = frames.shape[0]

    features = {
        'centroid_x_px': np.zeros((nframes,), 'float32'),
        'centroid_y_px': np.zeros((nframes,), 'float32'),
        'velocity_2d_px': np.zeros((nframes,), 'float32'),
        'velocity_3d_px': np.zeros((nframes,), 'float32'),
        'width_px': np.zeros((nframes,), 'float32'),
        'length_px': np.zeros((nframes,), 'float32'),
        'area_px': np.zeros((nframes,)),
        'centroid_x_mm': np.zeros((nframes,), 'float32'),
        'centroid_y_mm': np.zeros((nframes,), 'float32'),
        'velocity_2d_mm': np.zeros((nframes,), 'float32'),
        'velocity_3d_mm': np.zeros((nframes,), 'float32'),
        'width_mm': np.zeros((nframes,), 'float32'),
        'length_mm': np.zeros((nframes,), 'float32'),
        'area_mm': np.zeros((nframes,)),
        'height_ave_mm': np.zeros((nframes,), 'float32'),
        'angle': np.zeros((nframes,), 'float32'),
        'velocity_theta': np.zeros((nframes,)),
    }

    centroid_mm = convert_pxs_to_mm(track_features['centroid'], true_depth=true_depth)
    centroid_mm_shift = convert_pxs_to_mm(track_features['centroid'] + 1, true_depth=true_depth)

    px_to_mm = np.abs(centroid_mm_shift - centroid_mm)
    masked_frames = np.logical_and(frames > min_height, frames < max_height)

    features['centroid_x_px'] = track_features['centroid'][:, 0]
    features['centroid_y_px'] = track_features['centroid'][:, 1]

    features['centroid_x_mm'] = centroid_mm[:, 0]
    features['centroid_y_mm'] = centroid_mm[:, 1]

    # based on the centroid of the mouse, get the mm_to_px conversion

    features['width_px'] = np.min(track_features['axis_length'], axis=1)
    features['length_px'] = np.max(track_features['axis_length'], axis=1)
    features['area_px'] = np.sum(masked_frames, axis=(1, 2))

    features['width_mm'] = features['width_px'] * px_to_mm[:, 1]
    features['length_mm'] = features['length_px'] * px_to_mm[:, 0]
    features['area_mm'] = features['area_px'] * px_to_mm.mean(axis=1)

    features['angle'] = track_features['orientation']

    nmask = np.sum(masked_frames, axis=(1, 2))

    for i in range(nframes):
        if nmask[i] > 0:
            features['height_ave_mm'][i] = np.mean(
                frames[i, masked_frames[i, ...]])

    vel_x = np.diff(np.concatenate((features['centroid_x_px'][:1], features['centroid_x_px'])))
    vel_y = np.diff(np.concatenate((features['centroid_y_px'][:1], features['centroid_y_px'])))
    vel_z = np.diff(np.concatenate((features['height_ave_mm'][:1], features['height_ave_mm'])))

    features['velocity_2d_px'] = np.hypot(vel_x, vel_y)
    features['velocity_3d_px'] = np.sqrt(
        np.square(vel_x)+np.square(vel_y)+np.square(vel_z))

    vel_x = np.diff(np.concatenate((features['centroid_x_mm'][:1], features['centroid_x_mm'])))
    vel_y = np.diff(np.concatenate((features['centroid_y_mm'][:1], features['centroid_y_mm'])))

    features['velocity_2d_mm'] = np.hypot(vel_x, vel_y)
    features['velocity_3d_mm'] = np.sqrt(
        np.square(vel_x)+np.square(vel_y)+np.square(vel_z))

    features['velocity_theta'] = np.arctan2(vel_y, vel_x)

    return features


def feature_hampel_filter(features, centroid_hampel_span=None, centroid_hampel_sig=3,
                          angle_hampel_span=None, angle_hampel_sig=3):
    '''
    Filters computed extraction features using Hampel Filtering.
    Parameters
    ----------
    features (dict): dictionary of video features
    centroid_hampel_span (int): Centroid Hampel Span Filtering Kernel Size
    centroid_hampel_sig (int): Centroid Hampel Signal Filtering Kernel Size
    angle_hampel_span (int): Angle Hampel Span Filtering Kernel Size
    angle_hampel_sig (int): Angle Hampel Span Filtering Kernel Size

    Returns
    -------
    features (dict): filtered version of input dict.
    '''
    if centroid_hampel_span is not None and centroid_hampel_span > 0:
        padded_centroids = np.pad(features['centroid'],
                                  (((centroid_hampel_span // 2, centroid_hampel_span // 2)),
                                   (0, 0)),
                                  'constant', constant_values = np.nan)
        for i in range(1):
            vws = strided_app(padded_centroids[:, i], centroid_hampel_span, 1)
            med = np.nanmedian(vws, axis=1)
            mad = np.nanmedian(np.abs(vws - med[:, None]), axis=1)
            vals = np.abs(features['centroid'][:, i] - med)
            fill_idx = np.where(vals > med + centroid_hampel_sig * mad)[0]
            features['centroid'][fill_idx, i] = med[fill_idx]

        padded_orientation = np.pad(features['orientation'],
                                    (angle_hampel_span // 2, angle_hampel_span // 2),
                                    'constant', constant_values = np.nan)

    if angle_hampel_span is not None and angle_hampel_span > 0:
        vws = strided_app(padded_orientation, angle_hampel_span, 1)
        med = np.nanmedian(vws, axis=1)
        mad = np.nanmedian(np.abs(vws - med[:, None]), axis=1)
        vals = np.abs(features['orientation'] - med)
        fill_idx = np.where(vals > med + angle_hampel_sig * mad)[0]
        features['orientation'][fill_idx] = med[fill_idx]

    return features


def model_smoother(features, ll=None, clips=(-300, -125)):
    '''
    Spatial feature filtering.
    Parameters
    ----------
    features (dict): dictionary of extraction scalar features
    ll (np.array): list of loglikelihoods of pixels in frame
    clips (tuple): tuple to ensure video is indexed properly

    Returns
    -------
    features (dict) - smoothed version of input features
    '''

    if ll is None or clips is None or (clips[0] >= clips[1]):
        return features

    ave_ll = np.zeros((ll.shape[0], ))
    for i, ll_frame in enumerate(ll):

        max_mu = clips[1]
        min_mu = clips[0]

        smoother = np.mean(ll[i])
        smoother -= min_mu
        smoother /= (max_mu - min_mu)

        smoother = np.clip(smoother, 0, 1)
        ave_ll[i] = smoother

    for k, v in features.items():
        nans = np.isnan(v)
        ndims = len(v.shape)
        xvec = np.arange(len(v))
        if nans.any():
            if ndims == 2:
                for i in range(v.shape[1]):
                    f = scipy.interpolate.interp1d(xvec[~nans[:, i]], v[~nans[:, i], i],
                                                   kind='nearest', fill_value='extrapolate')
                    fill_vals = f(xvec[nans[:, i]])
                    features[k][xvec[nans[:, i]], i] = fill_vals
            else:
                f = scipy.interpolate.interp1d(xvec[~nans], v[~nans],
                                               kind='nearest', fill_value='extrapolate')
                fill_vals = f(xvec[nans])
                features[k][nans] = fill_vals

    for i in range(2, len(ave_ll)):
        smoother = ave_ll[i]
        for k, v in features.items():
            features[k][i] = (1 - smoother) * v[i - 1] + smoother * v[i]

    for i in reversed(range(len(ave_ll) - 1)):
        smoother = ave_ll[i]
        for k, v in features.items():
            features[k][i] = (1 - smoother) * v[i + 1] + smoother * v[i]

    return features
