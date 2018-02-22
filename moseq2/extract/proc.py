import moseq2.io.video
import moseq2.extract.roi
import numpy as np
import skimage.measure
import skimage.morphology
import scipy.stats
import scipy.signal
import cv2
import tqdm
import joblib
from copy import deepcopy


def get_flips(frames, flip_file=None, smoothing=None):
    """
    """

    try:
        clf = joblib.load(flip_file)
    except IOError:
        print("Could not open file {}".format(flip_file))
        raise

    flip_class = np.where(clf.classes_ == 1)[0]
    probas = clf.predict_proba(
        frames.reshape((-1, frames.shape[1]*frames.shape[2])))

    if smoothing:
        for i in range(probas.shape[1]):
            probas[:, i] = scipy.signal.medfilt(probas[:, i], smoothing)

    flips = probas.argmax(axis=1) == flip_class

    return flips


def get_largest_cc(frames, progress_bar=False):
    """
    Returns the largest connected component in an image
    """
    foreground_obj = np.zeros((frames.shape), 'bool')

    for i in tqdm.tqdm(range(frames.shape[0]), disable=not progress_bar, desc='CC'):
        nb_components, output, stats, centroids =\
            cv2.connectedComponentsWithStats(frames[i, ...], connectivity=4)
        szs = stats[:, -1]
        foreground_obj[i, ...] = output == szs[1:].argmax()+1

    return foreground_obj


def get_bground_im(frames):
    """
    Get background from frames
    """
    bground = np.median(frames, 0)
    return bground


def get_bground_im_file(frames_file, frame_stride=500, med_scale=5):
    """
    Get background from frames
    """

    finfo = moseq2.io.video.get_raw_info(frames_file)

    frame_idx = np.arange(0, finfo['nframes'], frame_stride)
    frame_store = np.zeros((len(frame_idx), finfo['dims'][1], finfo['dims'][0]))

    for i, frame in enumerate(frame_idx):
        frame_store[i, ...] = cv2.medianBlur(moseq2.io.video.read_frames_raw(
            frames_file, int(frame)).squeeze(), med_scale)

    bground = np.median(frame_store, 0)
    return bground


def get_bbox(roi):
    """
    Given a binary mask, return an array with the x and y boundaries
    """
    y, x = np.where(roi > 0)
    bbox = np.array([[y.min(), x.min()], [y.max(), x.max()]])
    return bbox


def get_roi(depth_image,
            strel_dilate=cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
            noise_tolerance=30,
            weights=(1, .1, 1),
            **kwargs):
    """
    Get an ROI using RANSAC plane fitting and simple blob features
    """

    roi_plane, dists = moseq2.extract.roi.plane_ransac(
        depth_image, noise_tolerance=noise_tolerance, **kwargs)
    dist_ims = dists.reshape(depth_image.shape)
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
    shape_index = np.mean(np.multiply(ranks.astype('float32'), weight_array[:,np.newaxis]), 0).argsort()

    # expansion microscopy on the roi

    rois = []
    bboxes = []

    for shape in shape_index:
        roi = np.zeros_like(depth_image)
        roi[region_properties[shape].coords[:, 0],
            region_properties[shape].coords[:, 1]] = 1
        roi = cv2.dilate(roi, strel_dilate, iterations=1)
        # roi=skimage.morphology.dilation(roi,dilate_element)
        rois.append(roi)
        bboxes.append(get_bbox(roi))

    return rois, bboxes, label_im, ranks, shape_index


def apply_roi(frames, roi):
    """
    Apply ROI to data, consider adding constraints (e.g. mod32==0)
    """
    # yeah so fancy indexing slows us down by 3-5x
    cropped_frames = frames*roi
    bbox = get_bbox(roi)
    # cropped_frames[:,roi==0]=0
    cropped_frames = cropped_frames[:, bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1]]
    return cropped_frames


def im_moment_features(IM):
    """
    Use the method of moments and centralized moments to get image properties

    Args:
        IM (2d numpy array): depth image

    Returns:
        Features (dictionary): returns a dictionary with orientation,
        centroid, and ellipse axis length

    """

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
                 strel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                 iterations=2,
                 strel_min=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                 iterations_min=None, progress_bar=True):
    """
    Simple filtering, median filter and morphological opening

    Args:
        frames (3d np array): frames x r x c
        strel (opencv structuring element): strel for morph opening
        iterations (int): number of iterations to run opening

    Returns:
        filtered_frames (3d np array): frame x r x c

    """
    # seeing enormous speed gains w/ opencv
    filtered_frames = deepcopy(frames).astype('uint8')

    for i in tqdm.tqdm(range(frames.shape[0]),
                       disable=not progress_bar, desc='Cleaning frames'):

        if iterations_min:
            filtered_frames[i, ...] = cv2.erode(filtered_frames[i, ...], strel_min, iterations_min)

        if prefilter_space:
            for j in range(len(prefilter_space)):
                filtered_frames[i, ...] = cv2.medianBlur(filtered_frames[i, ...], prefilter_space[j])

        if iterations:
            filtered_frames[i, ...] = cv2.morphologyEx(
                filtered_frames[i, ...], cv2.MORPH_OPEN, strel, iterations)

    if prefilter_time:
        for j in range(len(prefilter_time)):
            filtered_frames = scipy.signal.medfilt(
                filtered_frames, [prefilter_time[j], 1, 1])

    return filtered_frames


def get_frame_features(frames, frame_threshold=10, mask=np.array([]),
                       mask_threshold=-30, use_cc=False, progress_bar=True):
    """
    Use image moments to compute features of the largest object in the frame

    Args:
        frames (3d np array)
        frame_threshold (int): threshold in mm separating floor from mouse

    Returns:
        features (dict list): dictionary with simple image features

    """

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

    for i in tqdm.tqdm(range(nframes), disable=not progress_bar, desc='Computing moments'):

        frame_mask = frames[i, ...] > frame_threshold

        if use_cc:
            cc_mask = get_largest_cc((frames[[i], ...] > mask_threshold).astype('uint8')).squeeze()
            frame_mask = np.logical_and(cc_mask, frame_mask)

        if has_mask:
            frame_mask = np.logical_and(frame_mask, mask[i, ...] > mask_threshold)
        else:
            mask[i, ...] = frame_mask

        im2, cnts, hierarchy = cv2.findContours(
            (frame_mask).astype('uint8'),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tmp = np.array([cv2.contourArea(x) for x in cnts])

        if tmp.size == 0:
            continue

        mouse_cnt = tmp.argmax()

        for key, value in im_moment_features(cnts[mouse_cnt]).items():
            features[key][i] = value

    return features, mask


def crop_and_rotate_frames(frames, features, crop_size=(80, 80),
                           progress_bar=True):

    nframes = frames.shape[0]
    cropped_frames = np.zeros((nframes, 80, 80), frames.dtype)

    for i in tqdm.tqdm(range(frames.shape[0]), disable=not progress_bar, desc='Rotating'):

        if np.any(np.isnan(features['centroid'][i, :])):
            continue

        use_frame = np.pad(frames[i, ...], (crop_size, crop_size), 'constant', constant_values=0)

        rr = np.arange(features['centroid'][i, 1]-40, features['centroid'][i, 1]+41).astype('int16')
        cc = np.arange(features['centroid'][i, 0]-40, features['centroid'][i, 0]+41).astype('int16')

        rr = rr+crop_size[0]
        cc = cc+crop_size[1]

        if (np.any(rr >= use_frame.shape[0]) or np.any(rr < 1)
                or np.any(cc >= use_frame.shape[1]) or np.any(cc < 1)):
            continue

        rot_mat = cv2.getRotationMatrix2D((40, 40), -np.rad2deg(features['orientation'][i]), 1)
        cropped_frames[i, :, :] = cv2.warpAffine(use_frame[rr[0]:rr[-1], cc[0]:cc[-1]], rot_mat, (80, 80))

    return cropped_frames


def compute_scalars(frames, track_features, min_height=10, max_height=100):

    nframes = frames.shape[0]

    features = {
        'centroid_x': np.zeros((nframes,), 'float32'),
        'centroid_y': np.zeros((nframes,), 'float32'),
        'angle': np.zeros((nframes,), 'float32'),
        'width': np.zeros((nframes,), 'float32'),
        'length': np.zeros((nframes,), 'float32'),
        'height_ave': np.zeros((nframes,), 'float32'),
        'velocity_mag': np.zeros((nframes,), 'float32'),
        'velocity_theta': np.zeros((nframes,)),
        'area': np.zeros((nframes,)),
        'velocity_mag_3d': np.zeros((nframes,), 'float32'),
    }

    features['centroid_x'] = track_features['centroid'][:, 0]
    features['centroid_y'] = track_features['centroid'][:, 1]
    features['angle'] = track_features['orientation']
    features['width'] = np.min(track_features['axis_length'], axis=1)
    features['length'] = np.max(track_features['axis_length'], axis=1)
    masked_frames = np.logical_and(frames > min_height, frames < max_height)
    features['area'] = np.sum(masked_frames, axis=(1, 2))

    nmask = np.sum(masked_frames, axis=(1, 2))

    for i in range(nframes):
        if nmask[i] > 0:
            features['height_ave'][i] = np.mean(
                frames[i, masked_frames[i, ...]])

    vel_x = np.diff(np.pad(features['centroid_x'], (1, 0), 'edge'))
    vel_y = np.diff(np.pad(features['centroid_y'], (1, 0), 'edge'))
    vel_z = np.diff(np.pad(features['height_ave'], (1, 0), 'edge'))

    features['velocity_mag'] = np.hypot(vel_x, vel_y)
    features['velocity_mag_3d'] = np.sqrt(
        np.square(vel_x)+np.square(vel_y)+np.square(vel_z))
    features['velocity_theta'] = np.arctan2(vel_y, vel_x)

    return features
