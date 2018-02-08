from skimage.external import tifffile
import numpy as np
import json

def write_image(filename, image, scale=True, dtype='uint16', metadata={}):
    """Save image data, possibly with scale factor for easy display
    """

    metadata={}

    if scale:
        iinfo=np.iinfo(dtype)
        image=image.astype(dtype)
        scale_factor=np.floor(iinfo.max/image.max()).astype(dtype)
        scaled_image=image*scale_factor
        metadata={'scale_factor':str(scale_factor)}

    tifffile.imsave(filename, scaled_image, compress=0, metadata=metadata)

def read_image(filename, dtype='uint16', scale=True, scale_key='scale_factor'):
    """Load image data, possibly with scale factor...
    """

    with tifffile.TiffFile(filename) as tif:
        tmp=tif

    image=tmp.asarray()

    if scale:
        image_desc=json.loads(tmp.pages[0].tags['image_description'].as_str()[2:-1])
        scale_factor=int(image_desc[scale_key])
        image=image/scale_factor

    return image
