import os
import cv2
import tarfile
import datetime
import subprocess
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def get_raw_info(filename, bit_depth=16, frame_dims=(512, 424)):
    '''
    Gets info from a raw data file with specified frame dimensions and bit depth.

    Parameters
    ----------
    filename (string): name of raw data file
    bit_depth (int): bits per pixel (default: 16)
    frame_dims (tuple): wxh or hxw of each frame

    Returns
    -------
    file_info (dict): dictionary containing depth file metadata
    '''

    bytes_per_frame = (frame_dims[0] * frame_dims[1] * bit_depth) / 8

    if type(filename) is str:
        file_info = {
            'bytes': os.stat(filename).st_size,
            'nframes': int(os.stat(filename).st_size / bytes_per_frame),
            'dims': frame_dims,
            'bytes_per_frame': bytes_per_frame
        }
        if filename.endswith('.mkv'):
            try:
                vid = cv2.VideoCapture(filename)
                h, w, nframes = vid.get(cv2.CAP_PROP_FRAME_HEIGHT), \
                                vid.get(cv2.CAP_PROP_FRAME_WIDTH), \
                                vid.get(cv2.CAP_PROP_FRAME_COUNT)

                bytes_per_frame = (int(w) * int(h) * bit_depth) / 8

                file_info = {
                    'bytes': os.stat(filename).st_size,
                    'nframes': int(nframes),
                    'dims': (int(w), int(h)),
                    'bytes_per_frame': int(bytes_per_frame)
                }
            except:
                pass

    elif type(filename) is tarfile.TarInfo:
        file_info = {
            'bytes': filename.size,
            'nframes': int(filename.size / bytes_per_frame),
            'dims': frame_dims,
            'bytes_per_frame': bytes_per_frame
        }
    return file_info


def read_frames_raw(filename, frames=None, frame_dims=(512, 424), bit_depth=16, dtype="<i2", tar_object=None):
    '''
    Reads in data from raw binary file.

    Parameters
    ----------
    filename (string): name of raw data file
    frames (list or range): frames to extract
    frame_dims (tuple): wxh of frames in pixels
    bit_depth (int): bits per pixel (default: 16)
    tar_object (tarfile.TarFile): TarFile object, used for loading data directly from tgz

    Returns
    -------
    chunk (numpy ndarray): nframes x h x w
    '''

    vid_info = get_raw_info(filename, frame_dims=frame_dims, bit_depth=bit_depth)

    if type(frames) is int:
        frames = [frames]
    elif not frames or (type(frames) is range) and len(frames) == 0:
        frames = range(0, vid_info['nframes'])

    seek_point = np.maximum(0, frames[0]*vid_info['bytes_per_frame'])
    read_points = len(frames)*frame_dims[0]*frame_dims[1]

    dims = (len(frames), frame_dims[1], frame_dims[0])

    if type(tar_object) is tarfile.TarFile:
        with tar_object.extractfile(filename) as f:
            f.seek(int(seek_point))
            chunk = f.read(int(len(frames) * vid_info['bytes_per_frame']))
            chunk = np.frombuffer(chunk, dtype=np.dtype(dtype)).reshape(dims)
    else:
        with open(filename, "rb") as f:
            f.seek(int(seek_point))
            chunk = np.fromfile(file=f,
                                dtype=np.dtype(dtype),
                                count=read_points).reshape(dims)

    return chunk


# https://gist.github.com/hiwonjoon/035a1ead72a767add4b87afe03d0dd7b
def get_video_info(filename):
    '''
    Get dimensions of data compressed using ffv1, along with duration via ffmpeg.

    Parameters
    ----------
    filename (string): name of file

    Returns
    -------
    (dict): dictionary containing video file metadata
    '''

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
    try:
        return {'file': filename,
            'dims': (int(out[0]), int(out[1])),
            'fps': float(out[2].split('/')[0])/float(out[2].split('/')[1]),
            'nframes': int(out[3])}
    except:
        return {'file': filename,
            'dims': (int(out[0]), int(out[1])),
            'fps': float(out[2].split('/')[0])/float(out[2].split('/')[1]),
            'nframes': out[3]}

def convert_mkv_to_avi(filename):
    '''
    Converts Azure MKV video file format to AVI.

    Parameters
    ----------
    filename (str) path to mkv file to convert

    Returns
    -------
    outpath (str): path to converted AVI video file.
    '''

    outpath = os.path.join(os.path.dirname(filename),'proc/depth.avi')
    command = ['ffmpeg',
               '-i', filename,
               '-map', '0:0',
               '-vsync', '0',
               outpath]

    pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = pipe.communicate()

    return outpath


# simple command to pipe frames to an ffv1 file
def write_frames(filename, frames, threads=6, fps=30,
                 pixel_format='gray16le', codec='ffv1', close_pipe=True,
                 pipe=None, slices=24, slicecrc=1, frame_size=None, get_cmd=False, verbose=0):
    '''
    Write frames to avi file using the ffv1 lossless encoder

    Parameters
    ----------
    filename (str): path to file to write to.
    frames (np.ndarray): frames to write
    threads (int): number of threads to write video
    fps (int): frames per second
    pixel_format (str): format video color scheme
    codec (str): ffmpeg encoding-writer method to use
    close_pipe (bool): indicates to close the open pipe to video when done writing.
    pipe (subProcess.Pipe): pipe to currently open video file.
    slices (int): number of frame slices to write at a time.
    slicecrc (int): check integrity of slices
    frame_size (tuple): shape/dimensions of image.
    get_cmd (bool): indicates whether function should return ffmpeg command (instead of executing)
    verbose (bool): output progress.

    Returns
    -------
    pipe (subProcess.Pipe): indicates whether video writing is complete.
    '''

    # we probably want to include a warning about multiples of 32 for videos
    # (then we can use pyav and some speedier tools)

    if not frame_size and type(frames) is np.ndarray:
        frame_size = '{0:d}x{1:d}'.format(frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = '{0:d}x{1:d}'.format(frames[0], frames[1])

    command = ['ffmpeg',
               '-y',
               '-loglevel', 'fatal',
               '-framerate', str(fps),
               '-f', 'rawvideo',
               '-s', frame_size,
               '-pix_fmt', pixel_format,
               '-i', '-',
               '-an',
               '-vcodec', codec,
               '-threads', str(threads),
               '-slices', str(slices),
               '-slicecrc', str(slicecrc),
               '-r', str(fps),
               filename]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    disable = False
    if verbose == 0:
        disable = True
    for i in tqdm(range(frames.shape[0]), disable=disable):
        pipe.stdin.write(frames[i, ...].astype('uint16').tostring())

    if close_pipe:
        pipe.stdin.close()
        return None
    else:
        return pipe


def read_frames(filename, frames=range(0,), threads=6, fps=30,
                pixel_format='gray16le', frame_size=None,
                slices=24, slicecrc=1, get_cmd=False):
    '''
    Reads in frames from the .nut/.avi file using a pipe from ffmpeg.

    Parameters
    ----------
    filename (str): filename to get frames from
    frames (list or 1d numpy array): list of frames to grab
    threads (int): number of threads to use for decode
    fps (int): frame rate of camera in Hz
    pixel_format (str): ffmpeg pixel format of data
    frame_size (str): wxh frame size in pixels
    slices (int): number of slices to use for decode
    slicecrc (int): check integrity of slices
    get_cmd (bool): indicates whether function should return ffmpeg command (instead of executing).

    Returns
    -------
    video (3d numpy array):  frames x h x w
    '''

    if not filename.endswith('.mkv'):
        try:
            finfo = get_video_info(filename)
        except:
            finfo = get_raw_info(filename)
    else:
        finfo = get_raw_info(filename)
        tmp = get_video_info(filename)
        frame_size = tmp['dims']

    if frames is None or len(frames) == 0:
        finfo = get_video_info(filename)
        frames = np.arange(finfo['nframes']).astype('int16')

    if not frame_size:
        frame_size = finfo['dims']

    if not filename.endswith('.mkv'):
        command = [
            'ffmpeg',
            '-loglevel', 'fatal',
            '-ss', str(datetime.timedelta(seconds=frames[0]/fps)),
            '-i', filename,
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
    else:
        command = [
            'ffmpeg',
            '-loglevel', 'fatal',
            '-ss', str(datetime.timedelta(seconds=frames[0] / fps)),
            '-i', filename,
            '-map', '0:0',
            '-vframes', str(len(frames)),
            '-f', 'image2pipe',
            '-s', '{:d}x{:d}'.format(frame_size[0], frame_size[1]),
            '-pix_fmt', pixel_format,
            '-threads', str(threads),
            '-slices', str(slices),
            '-slicecrc', str(slicecrc),
            '-vsync', '0',
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
    '''
    Writes out a false-colored mp4 video.

    Parameters
    ----------
    filename (str): path to file to write to.
    frames (np.ndarray): frames to write
    threads (int): number of threads to write video
    fps (int): frames per second
    pixel_format (str): format video color scheme
    codec (str): ffmpeg encoding-writer method to use
    slices (int): number of frame slices to write at a time.
    slicecrc (int): check integrity of slices
    frame_size (tuple): shape/dimensions of image.
    depth_min (int): minimum mouse depth from floor in (mm)
    depth_max (int): maximum mouse depth from floor in (mm)
    get_cmd (bool): indicates whether function should return ffmpeg command (instead of executing)
    cmap (str): color map to use.
    pipe (subProcess.Pipe): pipe to currently open video file.
    close_pipe (bool): indicates to close the open pipe to video when done writing.
    frame_range (range()): frame indices to write on video

    Returns
    -------
    pipe (subProcess.Pipe): indicates whether video writing is complete.
    '''


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
    for i in tqdm(range(frames.shape[0]), disable=True, desc="Writing frames"):
        disp_img = frames[i, ...].copy().astype('float32')
        disp_img = (disp_img-depth_min)/(depth_max-depth_min)
        disp_img[disp_img < 0] = 0
        disp_img[disp_img > 1] = 1
        disp_img = np.delete(use_cmap(disp_img), 3, 2)*255
        if frame_range is not None:
            try:
                cv2.putText(disp_img, str(frame_range[i]), txt_pos, font, 1, white, 2, cv2.LINE_AA)
            except:
                pass
        pipe.stdin.write(disp_img.astype('uint8').tostring())

    if close_pipe:
        pipe.stdin.close()
        return None
    else:
        return pipe

def load_movie_data(filename, frames=None, frame_dims=(512, 424), bit_depth=16, **kwargs):
    """
    Reads in frames
    """

    try:
        if filename.lower().endswith('.dat'):
            frame_data = read_frames_raw(filename,
                                         frames=frames,
                                         frame_dims=frame_dims,
                                         bit_depth=bit_depth)
        elif filename.lower().endswith('.avi'):
            if type(frames) is int:
                frames = [frames]
            frame_data = read_frames(filename,
                                     frames)
        elif filename.lower().endswith('.mkv'):
            if type(frames) is int:
                frames = [frames]
            frame_data = read_frames(filename, frames)

    except AttributeError as e:
        frame_data = read_frames_raw(filename,
                                     frames=frames,
                                     frame_dims=frame_dims,
                                     bit_depth=bit_depth,
                                     **kwargs)
    return frame_data


def get_movie_info(filename, frame_dims=(512, 424), bit_depth=16):
    '''
    Returns dict of movie metadata.

    Parameters
    ----------
    filename (str): path to video file
    frame_dims (tuple): video dimensions
    bit_depth (int): integer indicating data type encoding

    Returns
    -------
    metadata (dict): dictionary containing video file metadata
    '''

    try:
        if filename.lower().endswith('.dat'):
            metadata = get_raw_info(filename, frame_dims=frame_dims, bit_depth=bit_depth)
        elif filename.lower().endswith('.avi'):
            metadata = get_video_info(filename)
        elif filename.lower().endswith('.mkv'):
            metadata = get_raw_info(filename, frame_dims=frame_dims, bit_depth=bit_depth)
    except AttributeError as e:
        metadata = get_raw_info(filename)

    return metadata

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