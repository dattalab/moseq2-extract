import cv2
import numpy as np
from copy import deepcopy
from moseq2_extract.extract.track import em_tracking, em_get_ll
from moseq2_extract.extract.proc import (crop_and_rotate_frames,
                                         clean_frames, apply_roi, get_frame_features,
                                         get_flips, compute_scalars, feature_hampel_filter,
                                         model_smoother)

# one stop shopping for taking some frames and doing stuff
def extract_chunk(chunk, use_em_tracker=False, prefilter_space=(3,),
                  prefilter_time=None,
                  iters_tail=1, iters_min=0,
                  strel_tail=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                  strel_min=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                  min_height=10, max_height=100,
                  mask_threshold=-20, use_cc=False,
                  bground=None, roi=None,
                  rho_mean=0, rho_cov=0,
                  tracking_ll_threshold=-100, tracking_segment=True,
                  tracking_init_mean=None, tracking_init_cov=None,
                  tracking_init_strel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                  flip_classifier=None, flip_smoothing=51,
                  frame_dtype='uint8',
                  progress_bar=True, crop_size=(80, 80), true_depth=673.1,
                  centroid_hampel_span=5, centroid_hampel_sig=3,
                  angle_hampel_span=5, angle_hampel_sig=3,
                  model_smoothing_clips=(-300, -150), tracking_model_init='raw',
                  **kwargs):
    '''
    This function looks for a mouse in background-subtracted frames from a chunk of depth video.
    It is called from the moseq2_extract.helpers.extract module.

    Parameters
    ----------
    chunk (3d np.ndarray): chunk to extract - (chunksize, height, width)
    use_em_tracker (bool): The EM tracker uses expectation-maximization to fit a 3D gaussian on a frame-by-frame
        basis to the mouse's body and determine if pixels are mouse vs cable.
    prefilter_space (tuple): spatial kernel size
    prefilter_time (tuple): temporal kernel size
    iters_tail (int): number of filtering iterations on mouse tail
    iters_min (int): minimum tail filtering filter kernel size
    strel_tail (cv2::StructuringElement - Ellipse): filtering kernel size to filter out mouse tail.
    strel_min (cv2::StructuringElement - Rectangle): filtering kernel size to filter mouse body in cable recording cases.
    min_height (int): minimum (mm) distance of mouse to floor.
    max_height (int): maximum (mm) distance of mouse to floor.
    mask_threshold (int): Threshold on log-likelihood to include pixels for centroid and angle calculation
    use_cc (bool): boolean to use connected components in cv2 structuring elements
    bground (np.ndarray): numpy array represented previously computed background
    roi (np.ndarray): numpy array represented previously computed roi
    rho_mean (int): smoothing parameter for the mean
    rho_cov (int): smoothing parameter for the covariance
    tracking_ll_threshold (float):  threshold for calling pixels a cable vs a mouse (usually between -16 to -12).
        If the log-likelihood falls below this value, pixels are considered cable.
    tracking_segment (bool): boolean for whether to use only the largest blob for EM updates.
    tracking_init_mean (float): Initialized mean value for EM Tracking
    tracking_init_cov (float): Initialized covariance value for EM Tracking
    tracking_init_strel (cv2::StructuringElement - Ellipse):
    flip_classifier (str): path to pre-selected flip classifier.
    flip_smoothing (int): amount of smoothing to use for flip classifier.
    frame_dtype (str): Data type for processed frames
    save_path: (str): Path to save extracted results
    progress_bar (bool): Display progress bar
    crop_size (tuple): size of the cropped mouse image.
    true_depth (float): previously computed detected true depth value.
    centroid_hampel_span (int): Hampel filter span kernel size
    centroid_hampel_sig (int):  Hampel filter standard deviation
    angle_hampel_span (int): Angle filter span kernel size
    angle_hampel_sig (int): Angle filter standard deviation
    model_smoothing_clips (tuple): Model smoothing clips
    tracking_model_init (str): Method for tracking model initialization

    Returns
    -------
    results: (3d np.ndarray) - (nframes, crop_height, crop_width)
    extracted cropped, oriented and centered RGB video chunk to be written to file.
    '''

    # if we pass bground or roi files, be sure to use 'em...
    if bground:
        chunk = (bground-chunk).astype(frame_dtype)

    if roi:
        chunk = apply_roi(chunk)

    # denoise the frames before we do anything else
    filtered_frames = clean_frames(chunk,
                                   prefilter_space=prefilter_space,
                                   prefilter_time=prefilter_time,
                                   iters_tail=iters_tail,
                                   strel_tail=strel_tail,
                                   iters_min=iters_min,
                                   strel_min=strel_min,
                                   frame_dtype=frame_dtype,
                                   progress_bar=progress_bar)

    # if we need it, compute the em parameters (for tracking in presence of occluders)
    if use_em_tracker:
        parameters = em_tracking(
            filtered_frames, chunk, rho_mean=rho_mean,
            rho_cov=rho_cov, progress_bar=progress_bar,
            ll_threshold=tracking_ll_threshold, segment=tracking_segment,
            init_mean=tracking_init_mean, init_cov=tracking_init_cov,
            init_strel=tracking_init_strel, init_method=tracking_model_init)
        ll = em_get_ll(filtered_frames, progress_bar=progress_bar, **parameters)
    else:
        ll = None
        parameters = None

    # now get the centroid and orientation of the mouse
    features, mask = get_frame_features(filtered_frames,
                                        frame_threshold=min_height, mask=ll,
                                        mask_threshold=mask_threshold,
                                        use_cc=use_cc,
                                        progress_bar=progress_bar)

    incl = ~np.isnan(features['orientation'])
    features['orientation'][incl] = np.unwrap(features['orientation'][incl] * 2) / 2

    features = feature_hampel_filter(features,
                                     centroid_hampel_span=centroid_hampel_span,
                                     centroid_hampel_sig=centroid_hampel_sig,
                                     angle_hampel_span=angle_hampel_span,
                                     angle_hampel_sig=angle_hampel_sig)

    if ll is not None:
        features = model_smoother(features,
                                  ll=ll,
                                  clips=model_smoothing_clips)

    # crop and rotate the frames
    cropped_frames = crop_and_rotate_frames(
        chunk, features, crop_size=crop_size, progress_bar=progress_bar)
    cropped_filtered_frames = crop_and_rotate_frames(
        filtered_frames, features, crop_size=crop_size, progress_bar=progress_bar)

    if use_em_tracker:
        use_parameters = deepcopy(parameters)
        use_parameters['mean'][:, 0] = crop_size[1] // 2
        use_parameters['mean'][:, 1] = crop_size[0] // 2
        mask = em_get_ll(cropped_frames, progress_bar=progress_bar, **use_parameters)
    else:
        mask = crop_and_rotate_frames(
            mask, features, crop_size=crop_size, progress_bar=progress_bar)

    if flip_classifier:
        flips = get_flips(cropped_frames, flip_classifier, flip_smoothing)
        for flip in np.where(flips)[0]:
            cropped_frames[flip, :] = np.rot90(cropped_frames[flip, :], k=2)
            cropped_filtered_frames[flip, :] = np.rot90(cropped_filtered_frames[flip, :], k=2)
            mask[flip, :] = np.rot90(mask[flip, :], k=2)
        features['orientation'][flips] += np.pi

    else:
        flips = None

    # todo: put in an option to compute scalars on raw or filtered

    scalars = compute_scalars(cropped_filtered_frames,
                              features,
                              min_height=min_height,
                              max_height=max_height,
                              true_depth=true_depth)

    results = {
        'depth_frames': cropped_frames,
        'mask_frames': mask,
        'scalars': scalars,
        'flips': flips,
        'parameters': parameters
    }

    return results