from moseq2.extract.proc import crop_and_rotate_frames,\
    clean_frames, apply_roi, get_frame_features,\
    get_flips, compute_scalars
from moseq2.extract.track import em_tracking, em_get_ll
import cv2
import os
import numpy as np

# one stop shopping for taking some frames and doing stuff


def extract_chunk(chunk, use_em_tracker=False, prefilter_space=(3,),
                  prefilter_time=None,
                  strel_iters=1, min_iters=0,
                  strel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                  strel_min=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                  min_height=10, max_height=100,
                  mask_threshold=-20, use_cc=False,
                  bground=None, roi=None,
                  rho_mean=0, rho_cov=0,
                  flip_classifier=None, flip_smoothing=51,
                  save_path=os.path.join(os.getcwd(), 'proc'),
                  progress_bar=True, crop_size=(80, 80)):

    # if we pass bground or roi files, be sure to use 'em...

    if bground:
        chunk = (bground-chunk).astype('uint8')

    if roi:
        chunk = apply_roi(chunk)

    # denoise the frames before we do anything else

    filtered_frames = clean_frames(chunk,
                                   prefilter_space=prefilter_space,
                                   prefilter_time=prefilter_time,
                                   iterations=strel_iters,
                                   strel=strel,
                                   iterations_min=min_iters,
                                   strel_min=strel_min,
                                   progress_bar=progress_bar)

    # if we need it, compute the em parameters
    # (for tracking in presence of occluders)

    ll = None

    if use_em_tracker:
        # print('Computing EM parameters...')
        parameters = em_tracking(
            filtered_frames, rho_mean=rho_mean,
            rho_cov=rho_cov, progress_bar=progress_bar)
        ll = em_get_ll(chunk, progress_bar=progress_bar, **parameters)

    # now get the centroid and orientation of the mouse

    # print('Getting centroid and orientation...')
    features, mask = get_frame_features(filtered_frames,
                                        frame_threshold=min_height, mask=ll,
                                        mask_threshold=mask_threshold,
                                        use_cc=use_cc,
                                        progress_bar=progress_bar)

    incl = ~np.isnan(features['orientation'])
    features['orientation'][incl] = np.unwrap(
        features['orientation'][incl]*2)/2

    # crop and rotate the frames

    # print('Cropping frames...')
    cropped_frames = crop_and_rotate_frames(
        chunk, features, crop_size=crop_size, progress_bar=progress_bar)
    cropped_filtered_frames = crop_and_rotate_frames(
        filtered_frames, features, crop_size=crop_size, progress_bar=progress_bar)
    mask = crop_and_rotate_frames(
        mask, features, crop_size=crop_size, progress_bar=progress_bar)

    if flip_classifier:
        # print('Fixing flips...')
        flips = get_flips(cropped_frames, flip_classifier, flip_smoothing)
        cropped_frames[flips, ...] = np.flip(
            cropped_frames[flips, ...], axis=2)
        mask[flips, ...] = np.flip(mask[flips, ...], axis=2)

    scalars = compute_scalars(cropped_filtered_frames,
                              features, min_height, max_height)

    results = {
        'depth_frames': cropped_frames,
        'mask_frames': mask,
        'scalars': scalars
    }

    return results
