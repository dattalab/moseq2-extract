from moseq2.extract.proc import crop_and_rotate_frames,\
 clean_frames, get_roi, apply_roi,get_bground_im_file, get_frame_features
from moseq2.extract.track import em_tracking, em_get_ll
import cv2

# one stop shopping for taking some frames and doing stuff
def extract_chunk(chunk,use_em_tracker=False,med_scale=3,strel_iters=2,min_iters=1,
                  strel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),
                  strel_min=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),
                  min_height=10,max_height=200,
                  mask_ll_threshold=-30,bground_file=None,roi_file=None):


    # if we pass bground or roi files, be sure to use 'em...

    print('Cleaning frames...')
    filtered_frames=clean_frames(chunk,
                             med_scale=med_scale,
                             iterations=strel_iters,
                             strel=strel,
                             iterations_min=min_iters,
                             strel_min=strel_min)

    ll=None

    if use_em_tracker:
        print('Computing EM parameters...')
        parameters=em_tracking(filtered_frames)
        ll=em_get_ll(chunk,**parameters)

    print('Getting centroid and orientation...')
    features=get_frame_features(filtered_frames,frame_threshold=min_height, mask=ll,
                                mask_threshold=mask_ll_threshold)
    print('Cropping frames...')
    cropped_frames=crop_and_rotate_frames(chunk,features)

    return cropped_frames
