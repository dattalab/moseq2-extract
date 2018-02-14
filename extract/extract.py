from moseq2.extract.proc import crop_and_rotate_frames,\
 clean_frames, get_roi, apply_roi,get_bground_im_file, get_frame_features,\
 get_bground_im, get_flips
from moseq2.extract.track import em_tracking, em_get_ll
from moseq2.io.image import read_image, write_image
import cv2
import os
import numpy as np

# one stop shopping for taking some frames and doing stuff
def extract_chunk(chunk,use_em_tracker=False,med_scale=3,strel_iters=2,min_iters=1,
                  strel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),
                  strel_min=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),
                  min_height=10,max_height=100,
                  mask_threshold=-15,use_cc=False,
                  bground_file=None,roi_file=None,
                  rho_mean=0,rho_cov=0,flip_classifier=None,flip_smoothing=51,
                  save_path=os.path.join(os.getcwd(),'proc')):


    # if we pass bground or roi files, be sure to use 'em...

    if bground_file and os.path.exists(bground_file):
        bground=read_image(bground_file)
        chunk=bground-chunk

    if roi_file and os.path.exists(roi_file):
        roi=read_image(roi_file)>0
        chunk=apply_roi(chunk,roi).astype('uint8')

    # denoise the frames before we do anything else

    print('Cleaning frames...')
    filtered_frames=clean_frames(chunk,
                             med_scale=med_scale,
                             iterations=strel_iters,
                             strel=strel,
                             iterations_min=min_iters,
                             strel_min=strel_min)

    # if we need it, compute the em parameters (for tracking in presence of occluders)

    ll=None

    if use_em_tracker:
        print('Computing EM parameters...')
        parameters=em_tracking(filtered_frames,rho_mean=rho_mean,rho_cov=rho_cov)
        ll=em_get_ll(chunk,**parameters)

    # now get the centroid and orientation of the mouse

    print('Getting centroid and orientation...')
    features , mask=get_frame_features(filtered_frames,frame_threshold=min_height, mask=ll,
                                mask_threshold=mask_threshold, use_cc=use_cc)

    features['orientation']=np.unwrap(features['orientation']*2)/2

    # crop and rotate the frames

    print('Cropping frames...')
    cropped_frames=crop_and_rotate_frames(chunk,features)
    mask=crop_and_rotate_frames(mask,features)

    if flip_classifier:
        print('Fixing flips...')
        flips=get_flips(cropped_frames,flip_classifier,flip_smoothing)
        cropped_frames[flips,...]=np.flip(cropped_frames[flips,...],axis=2)
        mask[flips,...]=np.flip(mask[flips,...],axis=2)

    return cropped_frames,features,mask
