# import moseq2_extract.extract.proc
import numpy as np
import tqdm
import subprocess
import matplotlib.pyplot as plt
import os
import datetime
import cv2


def get_raw_info(filename, bit_depth=16, frame_dims=(512, 424)):
    """
    Gets info from a raw data file with specified frame dimensions and bit depth

    Args:
        filename (string): name of raw data file
        bit_depth (int): bits per pixel (default: 16)
        frame_dims (tuple): wxh or hxw of each frame
    """

    bytes_per_frame = (frame_dims[0]*frame_dims[1]*bit_depth)/8

    file_info = {
        'bytes': os.stat(filename).st_size,
        'nframes': int(os.stat(filename).st_size/bytes_per_frame),
        'dims': frame_dims,
        'bytes_per_frame': bytes_per_frame
    }

    return file_info


def read_frames_raw(filename, frames=None, frame_dims=(512, 424), bit_depth=16, dtype="<i2"):
    """
    Reads in data from raw binary file

    Args:
        filename (string): name of raw data files
        frames (list or range): frames to extract
        frame_dims (tuple): wxh of frames in pixels
        bit_depth (int): bits per pixel (default: 16)

    Returns:
        frames (numpy ndarray): frames x h x w
    """

    vid_info = get_raw_info(filename, frame_dims=frame_dims, bit_depth=bit_depth)

    if type(frames) is int:
        frames = [frames]
    elif not frames or (type(frames) is range) and len(frames) == 0:
        frames = range(0, vid_info['nframes'])

    seek_point = np.maximum(0, frames[0]*vid_info['bytes_per_frame'])
    read_points = len(frames)*frame_dims[0]*frame_dims[1]
    dims = (len(frames), frame_dims[1], frame_dims[0])

    with open(filename, "rb") as f:
        f.seek(seek_point.astype('int'))
        chunk = np.fromfile(file=f,
                            dtype=np.dtype(dtype),
                            count=read_points).reshape(dims)

    return chunk


# https://gist.github.com/hiwonjoon/035a1ead72a767add4b87afe03d0dd7b
def get_video_info(filename):
    """
    Get dimensions of data compressed using ffv1, along with duration via ffmpeg

    Args:
        filename (string): name of file
    """
    command = ['ffprobe',
               '-v', 'fatal',
               '-show_entries',
               'stream=width,height,r_frame_rate,nb_frames',
               '-of',
               'default=noprint_wrappers=1:nokey=1',
               filename,
               '-sexagesimal']

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()

    if(err):
        print(err)
    out = out.decode().split('\n')

    return {'file': filename,
            'dims': (int(out[0]), int(out[1])),
            'fps': float(out[2].split('/')[0])/float(out[2].split('/')[1]),
            'nframes': int(out[3])}


# simple command to pipe frames to an ffv1 file
def write_frames(filename, frames, threads=6, fps=30,
                 pixel_format='gray16le', codec='ffv1', close_pipe=True,
                 slices=24, slicecrc=1, frame_size=None, get_cmd=False):
    """
    Write frames to avi file using the ffv1 lossless encoder
    """

    # we probably want to include a warning about multiples of 32 for videos
    # (then we can use pyav and some speedier tools)

    if not frame_size and type(frames) is np.ndarray:
        frame_size = '{0:d}x{1:d}'.format(frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = '{0:d}x{1:d}'.format(frames[0], frames[1])

    command = ['ffmpeg',
               '-y',
               '-threads', str(threads),
               '-loglevel', 'fatal',
               '-framerate', str(fps),
               '-f', 'rawvideo',
               '-s', frame_size,
               '-pix_fmt', pixel_format,
               '-i', '-',
               '-an',
               '-vcodec', codec,
               '-slices', str(slices),
               '-slicecrc', str(slicecrc),
               '-r', str(fps),
               filename]

    if get_cmd:
        return command

    pipe = subprocess.Popen(
        command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in tqdm.tqdm(range(frames.shape[0])):
        pipe.stdin.write(frames[i, ...].astype('uint16').tostring())

    if close_pipe:
        pipe.stdin.close()
        return None
    else:
        return pipe


def read_frames(filename, frames=range(0,), threads=6, fps=30,
                pixel_format='gray16le', frame_size=None,
                slices=24, slicecrc=1, get_cmd=False):
    """
    Reads in frames from the .nut/.avi file using a pipe from ffmpeg.

    Args:
        filename (str): filename to get frames from
        frames (list or 1d numpy array): list of frames to grab
        threads (int): number of threads to use for decode
        fps (int): frame rate of camera in Hz
        pixel_format (str): ffmpeg pixel format of data
        frame_size (str): wxh frame size in pixels
        slices (int): number of slices to use for decode
        slicecrc (int): check integrity of slices

    Returns:
        3d numpy array:  frames x h x w
    """

    finfo = get_video_info(filename)

    if frames is None or len(frames) == 0:
        finfo = get_video_info(filename)
        frames = np.arange(finfo['nframes']).astype('int16')

    if not frame_size:
        frame_size = finfo['dims']

    command = [
        'ffmpeg',
        '-loglevel', 'fatal',
        '-i', filename,
        '-ss', str(datetime.timedelta(seconds=frames[0]/fps)),
        '-vframes', str(len(frames)),
        '-f', 'image2pipe',
        '-s', '{:d}x{:d}'.format(frame_size[0], frame_size[1]),
        '-pix_fmt', pixel_format,
        '-threads', str(threads),
        '-slices', str(slices),
        '-slicecrc', str(slicecrc),
        '-vcodec', 'rawvideo',
        '-'
    ]

    if get_cmd:
        return command

    pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = pipe.communicate()
    if(err):
        print('error', err)
        return None
    video = np.frombuffer(out, dtype='uint16').reshape((len(frames), frame_size[1], frame_size[0]))
    return video

# simple command to pipe frames to an ffv1 file


def write_frames_preview(filename, frames=np.empty((0,)), threads=6,
                         fps=30, pixel_format='rgb24',
                         codec='h264', slices=24, slicecrc=1,
                         frame_size=None, depth_min=0, depth_max=80,
                         get_cmd=False, cmap='jet',
                         pipe=None, close_pipe=True, frame_range=None):
    """
    Writes out a false-colored mp4 video
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    txt_pos = (5, frames.shape[-1] - 40)

    if not np.mod(frames.shape[1], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 1), (0, 0)), 'constant', constant_values=0)

    if not np.mod(frames.shape[2], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=0)

    if not frame_size and type(frames) is np.ndarray:
        frame_size = '{0:d}x{1:d}'.format(frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = '{0:d}x{1:d}'.format(frames[0], frames[1])

    command = ['ffmpeg',
               '-y',
               '-loglevel', 'fatal',
               '-threads', str(threads),
               '-framerate', str(fps),
               '-f', 'rawvideo',
               '-s', frame_size,
               '-pix_fmt', pixel_format,
               '-i', '-',
               '-an',
               '-vcodec', codec,
               '-slices', str(slices),
               '-slicecrc', str(slicecrc),
               '-r', str(fps),
               '-pix_fmt', 'yuv420p',
               filename]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # scale frames d00d

    use_cmap = plt.get_cmap(cmap)

    for i in tqdm.tqdm(range(frames.shape[0]), desc="Writing frames"):
        disp_img = frames[i, ...].copy().astype('float32')
        disp_img = (disp_img-depth_min)/(depth_max-depth_min)
        disp_img[disp_img < 0] = 0
        disp_img[disp_img > 1] = 1
        disp_img = np.delete(use_cmap(disp_img), 3, 2)*255
        if frame_range is not None:
            cv2.putText(disp_img, str(frame_range[i]), txt_pos, font, 1, white, 2, cv2.LINE_AA)
        pipe.stdin.write(disp_img.astype('uint8').tostring())

    if close_pipe:
        pipe.stdin.close()
        return None
    else:
        return pipe


# def encode_raw_frames_chunk(src_filename, bground_im, roi, bbox,
#                             chunk_size=1000, overlap=0, depth_min=5,
#                             depth_max=100,
#                             bytes_per_frame=int((424*512*16)/8)):
#
#     save_dir = os.path.join(os.path.dirname(src_filename), '_chunks')
#
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     base_filename = os.path.splitext(os.path.basename(src_filename))[0]
#
#     file_bytes = os.stat(src_filename).st_size
#     file_nframes = int(file_bytes/bytes_per_frame)
#     steps = np.append(np.arange(0, file_nframes, chunk_size), file_nframes)
#
#     # need to write out a manifest so we know the location of every frame
#     dest_filename = []
#
#     for i in tqdm.tqdm(range(steps.shape[0]-1)):
#         if i == 1:
#             chunk = read_frames_raw(src_filename, np.arange(steps[i], steps[i+1]))
#         else:
#             chunk = read_frames_raw(src_filename, np.arange(steps[i]-overlap, steps[i+1]))
#
#         chunk = (bground_im-chunk).astype('uint8')
#         chunk[chunk < depth_min] = 0
#         chunk[chunk > depth_max] = 0
#         chunk = moseq2_extract.extract.proc.apply_roi(chunk, roi, bbox)
#
#         dest_filename.append(os.path.join(save_dir, base_filename+'chunk{:05d}.avi'.format(i)))
#         write_frames(dest_filename[-1], chunk)
#
#     return dest_filename


def load_movie_data(filename, frames=None, frame_dims=(512, 424), bit_depth=16):
    """
    Reads in frames
    """
    if filename.lower().endswith('.dat'):
        frame_data = read_frames_raw(filename, frames=frames,
                                     frame_dims=frame_dims, bit_depth=bit_depth)
    elif filename.lower().endswith('.avi'):
        frame_data = read_frames(filename, frames)

    return frame_data


def get_movie_info(filename, frame_dims=(512, 424), bit_depth=16):
    """
    Gets movie info
    """
    if filename.lower().endswith('.dat'):
        metadata = get_raw_info(filename, frame_dims=frame_dims, bit_depth=bit_depth)
    elif filename.lower().endswith('.avi'):
        metadata = get_video_info(filename)

    return metadata
