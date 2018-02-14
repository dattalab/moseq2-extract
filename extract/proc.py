import moseq2.io.video
import moseq2.extract.roi
import numpy as np
import skimage.measure
import skimage.morphology
import scipy.stats
import cv2
import os
import tqdm
import pdb
import joblib
from copy import copy,deepcopy


#def classify_flips(frame,flip_file=None,)

def get_largest_cc(frame,progress_bar=False):
    """Returns the largest connected component in an image
    """
    foreground_obj=np.zeros((frame.shape),'bool')

    for i in tqdm.tqdm(range(frame.shape[0]),disable=not progress_bar):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(frame[i,...], connectivity=4)
        szs=stats[:,-1]
        foreground_obj[i,...]=output==szs[1:].argmax()+1

    return foreground_obj

def get_bground_im(frames):
    """Get background from frames
    """
    bground=np.median(frames,0)
    return bground

def get_bground_im_file(frames_file,frame_stride=500,med_scale=5):
    """Get background from frames
    """

    finfo=moseq2.io.video.get_raw_info(frames_file)

    frame_idx=np.arange(0,finfo['nframes'],frame_stride)
    frame_store=np.zeros((len(frame_idx),finfo['dims'][0],finfo['dims'][1]))

    for i,frame in enumerate(frame_idx):
        frame_store[i,...]=cv2.medianBlur(moseq2.io.video.read_frames_raw(frames_file,int(frame)).squeeze(),med_scale)

    bground=np.median(frame_store,0)
    return bground

def get_bbox(roi):
    """Given a binary mask, return an array with the x and y boundaries
    """
    y,x = np.where(roi>0)
    bbox=np.array([[y.min(),x.min()],[y.max(),x.max()]])
    return bbox

def get_roi(depth_image,strel_dilate=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30)),
            noise_tolerance=30,nrois=1,**kwargs):
    """Get an ROI using RANSAC plane fitting and simple blob features
    """

    roi_plane , dists=moseq2.extract.roi.plane_ransac(depth_image,noise_tolerance=noise_tolerance,**kwargs)
    dist_ims=dists.reshape(depth_image.shape)
    bin_im=dist_ims<noise_tolerance

    # anything < noise_tolerance from the plane is part of it

    label_im=skimage.measure.label(bin_im)
    region_properties=skimage.measure.regionprops(label_im)

    areas=np.zeros((len(region_properties),));
    extents=np.zeros_like(areas);
    dists=np.zeros_like(extents);

    # get the max distance from the center, area and extent

    center=np.array(depth_image.shape)/2

    for i,props in enumerate(region_properties):
        areas[i]=props.area
        extents[i]=props.extent
        tmp_dists=np.sqrt(np.sum(np.square(props.coords-center),1))
        dists[i]=tmp_dists.max()

    # rank features

    ranks=np.vstack((scipy.stats.rankdata(-areas,method='average'),
                     scipy.stats.rankdata(-extents,method='average'),
                     scipy.stats.rankdata(dists,method='average')))
    shape_index=np.mean(ranks.astype('float32')*np.array([[1],[.5],[1]]),0).argsort()

    # expansion microscopy on the roi

    rois=[]
    bboxes=[]

    for shape in shape_index:
        roi=np.zeros_like(depth_image)
        roi[region_properties[shape].coords[:,0],region_properties[shape].coords[:,1]]=1
        roi=cv2.dilate(roi,strel_dilate,iterations=1)
        #roi=skimage.morphology.dilation(roi,dilate_element)
        rois.append(roi)
        bboxes.append(get_bbox(roi))

    return rois , bboxes, label_im, ranks, shape_index


def apply_roi(frames,roi):
    """Apply ROI to data, consider adding constraints (e.g. mod32==0)
    """
    # yeah so fancy indexing slows us down by 3-5x
    cropped_frames=frames*roi
    bbox=get_bbox(roi)
    #cropped_frames[:,roi==0]=0
    cropped_frames=cropped_frames[:,bbox[0,0]:bbox[1,0],bbox[0,1]:bbox[1,1]]
    return cropped_frames


def im_moment_features(IM):
    """Use the method of moments and centralized moments to get image properties

    Args:
        IM (2d numpy array): depth image

    Returns:
        Features (dictionary): returns a dictionary with orientation, centroid, and ellipse axis length

    """

    tmp=cv2.moments(IM)
    num=2*tmp['mu11']
    den=tmp['mu20']-tmp['mu02']

    common=np.sqrt(4*np.square(tmp['mu11'])+np.square(den))

    features={
        'orientation':-.5*np.arctan2(num,den),
        'centroid':[tmp['m10']/tmp['m00'],tmp['m01']/tmp['m00']],
        'axis_length':[2*np.sqrt(2)*np.sqrt((tmp['mu20']+tmp['mu02']+common)/tmp['m00']),
                       2*np.sqrt(2)*np.sqrt((tmp['mu20']+tmp['mu02']-common)/tmp['m00'])]
    }

    return features


def clean_frames(frames,med_scale=3,
                 strel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),
                 iterations=2,
                 strel_min=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),
                 iterations_min=None):
    """Simple filtering, median filter and morphological opening

    Args:
        frames (3d np array): frames x r x c
        strel (opencv structuring element): strel for morph opening
        iterations (int): number of iterations to run opening

    Returns:
        filtered_frames (3d np array): frame x r x c

    """
    # seeing enormous speed gains w/ opencv
    filtered_frames=deepcopy(frames).astype('uint8')

    for i in tqdm.tqdm(range(frames.shape[0])):

        if iterations_min:
            filtered_frames[i,...]=cv2.erode(filtered_frames[i,...],strel_min,iterations_min)

        if med_scale:
            filtered_frames[i,...]=cv2.medianBlur(filtered_frames[i,...],med_scale)

        if iterations:
            filtered_frames[i,...]=cv2.morphologyEx(filtered_frames[i,...],cv2.MORPH_OPEN,strel,iterations)

    return filtered_frames

def get_frame_features(frames,frame_threshold=10,mask=np.array([]),mask_threshold=-30,use_cc=False):
    """Use image moments to compute features of the largest object in the frame

    Args:
        frames (3d np array)
        frame_threshold (int): threshold in mm separating floor from mouse

    Returns:
        features (dict list): dictionary with simple image features

    """

    features=[]

    if type(mask) is np.ndarray and mask.size>0:
        has_mask=True
    else:
        has_mask=False
        mask=np.zeros((frames.shape),'uint8')

    for i in tqdm.tqdm(range(frames.shape[0])):

        frame_mask=frames[i,...]>frame_threshold

        if use_cc:
            cc_mask=get_largest_cc((frames[[i],...]>mask_threshold).astype('uint8')).squeeze()
            frame_mask=np.logical_and(cc_mask,frame_mask)

        if has_mask:
            frame_mask=np.logical_and(frame_mask,mask[i,...]>mask_threshold)
        else:
            mask[i,...]=frame_mask

        im2,cnts,hierarchy=cv2.findContours((frame_mask).astype('uint8'),
                                            cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        tmp=np.array([cv2.contourArea(x) for x in cnts])
        mouse_cnt=tmp.argmax()
        features.append(im_moment_features(cnts[mouse_cnt]))

    return features, mask


def crop_and_rotate_frames(frames,features,crop_size=(80,80)):

    nframes=frames.shape[0]
    cropped_frames=np.zeros((nframes,80,80),frames.dtype)

    #padded_frames=np.pad(frames,((0,0),crop_size,crop_size),'constant',constant_values=0)

    for i in tqdm.tqdm(range(frames.shape[0])):

        use_frame=np.pad(frames[i,...],(crop_size,crop_size),'constant',constant_values=0)

        rr=np.arange(features[i]['centroid'][1]-40,features[i]['centroid'][1]+41).astype('int16')
        cc=np.arange(features[i]['centroid'][0]-40,features[i]['centroid'][0]+41).astype('int16')

        rr=rr+crop_size[0]
        cc=cc+crop_size[1]

        if np.any(rr>=use_frame.shape[0]) or np.any(rr<1) or np.any(cc>=use_frame.shape[1]) or np.any(cc<1):
            continue

        rot_mat=cv2.getRotationMatrix2D((40,40),-np.rad2deg(features[i]['orientation']),1)
        cropped_frames[i,:,:]=cv2.warpAffine(use_frame[rr[0]:rr[-1],cc[0]:cc[-1]],rot_mat,(80,80))

    return cropped_frames
