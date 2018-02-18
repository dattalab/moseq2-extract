from skimage.external import tifffile
import numpy as np
import json
import os
import ast

def write_image(filename, image, scale=True, scale_factor=None, dtype='uint16', metadata={}):
    """Save image data, possibly with scale factor for easy display
    """

    metadata={}

    if scale:
        iinfo=np.iinfo(dtype)
        image=image.astype(dtype)

        if not scale_factor:
            scale_factor=np.floor(iinfo.max/image.max()).astype(dtype)
            image=image*scale_factor
        elif type(scale_factor) is tuple:
            image=image.astype('float32')
            image=(image-scale_factor[0])/(scale_factor[1]-scale_factor[0])
            image[image<0]=0
            image[image>1]=1
            image=image*iinfo.max
            image=image.astype(dtype)

        metadata={'scale_factor':str(scale_factor)}

    directory=os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    tifffile.imsave(filename, image, compress=0, metadata=metadata)

def read_image(filename, dtype='uint16', scale=True, scale_key='scale_factor'):
    """Load image data, possibly with scale factor...
    """

    with tifffile.TiffFile(filename) as tif:
        tmp=tif

    image=tmp.asarray()

    if scale:
        image_desc=json.loads(tmp.pages[0].tags['image_description'].as_str()[2:-1])

        try:
            scale_factor=int(image_desc[scale_key])
        except:
            scale_factor=ast.literal_eval(image_desc[scale_key])

        if type(scale_factor) is int:
            image=image/scale_factor
        elif type(scale_factor) is tuple:
            iinfo=np.iinfo(image.dtype)
            image=image.astype('float32')/iinfo.max
            image=image*(scale_factor[1]-scale_factor[0])+scale_factor[0]

    return image
