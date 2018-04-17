import numpy as np
import os
import json
import cv2


def gen_batch_sequence(nframes, chunk_size, overlap):
    """Generate a sequence with overlap
    """
    seq = range(nframes)
    for i in range(0, len(seq)-overlap, chunk_size-overlap):
        yield seq[i:i+chunk_size]


def load_timestamps(timestamp_file, col=0):
    """Read timestamps from space delimited text file
    """

    ts = []
    with open(timestamp_file, 'r') as f:
        for line in f:
            cols = line.split()
            ts.append(float(cols[col]))

    return np.array(ts)


def load_metadata(metadata_file):

    metadata = {}

    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    return metadata


def select_strel(string='e', size=(10, 10)):
    if string[0].lower() == 'e':
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif string[0].lower() == 'r':
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    return strel


def recursive_find_unextracted_dirs(root_dir=os.getcwd(),
                               ext='.dat',
                               yaml_path='/proc/results.yaml',
                               metadata_path='metadata.json'):
    """Recursively find unextracted directories
    """
    proc_dirs = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            status_file = os.path.join(root, yaml_path)
            metadata_file = os.path.join(root, metadata_path)
            if file.endswith(ext) and not os.path.exists(status_file) and os.path.exists(metadata_file):
                proc_dirs.append(os.path.join(root, file))
    return proc_dirs
