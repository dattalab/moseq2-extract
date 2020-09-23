import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from moseq2_extract.util import scalar_attributes

def check_timestamp_error_percentage(timestamps, fps):
    '''
    https://www.mathworks.com/help/imaq/examples/determining-the-rate-of-acquisition.html

    Parameters
    ----------
    timestamps
    fps

    Returns
    -------
    percentError

    '''


    # Find the time difference between frames.
    diff = np.diff(timestamps) / 1000

    # Find the average time difference between frames.
    avgTime = np.mean(diff)

    # Determine the experimental frame rate.
    expRate = 1 / avgTime

    # Determine the percent error between the determined and actual frame rate.
    diffRates = abs(fps - expRate)
    percentError = (diffRates / fps) * 100

    return percentError

def count_nan_rows(scalar_df):
    '''

    Parameters
    ----------
    scalar_df

    Returns
    -------
    n_missing_frames
    '''

    nanrows = scalar_df.isnull().sum(axis=1).to_numpy()

    n_missing_frames = len(nanrows[nanrows > 0])

    return n_missing_frames

def count_missing_mouse_frames(scalar_df):
    '''

    Parameters
    ----------
    scalar_df

    Returns
    -------
    missing_mouse_frames
    '''

    missing_mouse_frames = len(scalar_df[scalar_df['area_px'] == 0])

    return missing_mouse_frames

# warning: min height may be too high
def count_frames_with_small_areas(scalar_df):
    '''

    Parameters
    ----------
    scalar_df

    Returns
    -------
    corrupt_frames
    '''

    corrupt_frames = len(scalar_df[scalar_df['area_px'] < scalar_df['area_px'].std()])

    return corrupt_frames

def count_stationary_frames(scalar_df):
    '''

    Parameters
    ----------
    scalar_df

    Returns
    -------
    motionless_frames
    '''

    motionless_frames = len(scalar_df[scalar_df['velocity_2d_mm'] < 0.1])-1 # subtract 1 because first frame is always 0mm/s

    return motionless_frames