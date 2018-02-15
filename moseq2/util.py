import numpy as np
import os
import json

def gen_batch_sequence(nframes,chunk_size,overlap):
    """Generate a sequence with overlap
    """
    seq=range(nframes)
    for i in range(0,len(seq)-overlap,chunk_size-overlap):
        yield seq[i:i+chunk_size]

def load_timestamps(timestamp_file,col=0):
    """Read timestamps from space delimited text file
    """

    ts=[]
    with open(timestamp_file,'r') as f:
        for line in f:
            cols=line.split()
            ts.append(float(cols[col]))

    return np.array(ts)

def load_metadata(metadata_file):

    metadata={}

    if os.path.exists(metadata_file):
        with open(metadata_file,'r') as f:
            metadata=json.load(f)

    return metadata
