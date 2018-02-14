from moseq2.extract.proc import crop_and_rotate_frames,\
 clean_frames, get_roi, apply_roi,get_bground_im_file, get_frame_features,get_bground_im
from moseq2.extract.track import em_tracking, em_get_ll
from moseq2.io.image import read_image, write_image
import cv2
import os
import numpy as np

# one stop shopping for taking some frames and doing stuff
def extract_chunk(chunk,use_em_tracker=False,med_scale=3,strel_iters=2,min_iters=1,
                  strel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),
                  strel_min=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),
                  min_height=10,max_height=200,
                  mask_threshold=-30,use_cc=False,
                  bground_file=None,roi_file=None,
                  rho_mean=0,rho_cov=0,flip_classifier=None,
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

    tmp=np.unwrap(np.array([x['orientation'] for x in features])*2)/2

    for i in range(tmp.shape[0]):
        features[i]['orientation']=tmp[i]

    # crop and rotate the frames

    print('Cropping frames...')
    cropped_frames=crop_and_rotate_frames(chunk,features)
    mask=crop_and_rotate_frames(mask,features)

    #if flip_classifier:


    return cropped_frames,features,mask
