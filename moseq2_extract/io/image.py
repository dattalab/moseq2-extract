from skimage.external import tifffile
import numpy as np
import json
import os
import ast
from pathlib import Path


def write_image(filename, image, scale=True,
                scale_factor=None, dtype='uint16',
                metadata={}, compress=0):
    """Save image data, possibly with scale factor for easy display
    """
    file = Path(filename)

    metadata = {}

    if scale:
        max_int = np.iinfo(dtype).max
        image = image.astype(dtype)

        if not scale_factor:
            # scale image to `dtype`'s full range
            scale_factor = int(max_int / np.nanmax(image))
            image = image * scale_factor
        elif isinstance(scale_factor, tuple):
            image = np.float32(image)
            image = (image - scale_factor[0]) / (scale_factor[1] - scale_factor[0])
            image = np.clip(image, 0, 1) * max_int

        metadata = {'scale_factor': str(scale_factor)}

    directory = file.parent
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    tifffile.imsave(file.as_posix(), image.astype(dtype), compress=compress, metadata=metadata)


def read_image(filename, dtype='uint16', scale=True, scale_key='scale_factor'):
    """Load image data, possibly with scale factor...
    """

    with tifffile.TiffFile(filename) as tif:
        tmp = tif

    image = tmp.asarray()

    if scale:
        image_desc = json.loads(
            tmp.pages[0].tags['image_description'].as_str()[2:-1])

        try:
            scale_factor = int(image_desc[scale_key])
        except ValueError:
            scale_factor = ast.literal_eval(image_desc[scale_key])

        if type(scale_factor) is int:
            image = image/scale_factor
        elif type(scale_factor) is tuple:
            iinfo = np.iinfo(image.dtype)
            image = image.astype('float32')/iinfo.max
            image = image*(scale_factor[1]-scale_factor[0])+scale_factor[0]

    return image
