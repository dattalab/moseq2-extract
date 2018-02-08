import numpy as np
import tqdm
import subprocess
import matplotlib.pyplot as plt
import os
from copy import deepcopy

def get_raw_info(filename,bytes_per_frame=int((424*512*16)/8),frame_dims=[424,512]):

    file_info={
        'bytes':os.stat(filename).st_size,
        'nframes':int(os.stat(filename).st_size/bytes_per_frame),
        'dims':frame_dims
    }

    return file_info

def read_frames_raw(filename,frames,frame_dims=(424,512)):
    """Read in binary data
    """
    if type(frames) is int:
        frames=[frames]

    bytes_per_frame=int((frame_dims[0]*frame_dims[1]*16)/8)
    seek_point=np.maximum(0,frames[0]*bytes_per_frame)
    read_points=len(frames)*frame_dims[0]*frame_dims[1]

    with open(filename,"rb") as f:
        f.seek(seek_point)
        chunk = np.fromfile(file=f,dtype=np.dtype("<i2"),\
                            count=read_points).reshape(len(frames),frame_dims[0],frame_dims[1])

    return chunk


# https://gist.github.com/hiwonjoon/035a1ead72a767add4b87afe03d0dd7b
def get_video_info(filename):
    """Get dimensions of data compressed using ffv1, along with duration
    """
    command = ['ffprobe',
               '-v', 'fatal',
               '-show_entries',
               'stream=width,height,r_frame_rate,nb_frames',
               '-of',
               'default=noprint_wrappers=1:nokey=1',
               filename,
               '-sexagesimal']
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print(err)
    out = out.decode().split('\n')
    return {'file' : filename,
            'width': int(out[0]),
            'height' : int(out[1]),
            'fps': float(out[2].split('/')[0])/float(out[2].split('/')[1]),
            'duration' : int(out[3]) }


# simple command to pipe frames to an ffv1 file
def write_frames(filename,frames,threads=6,camera_fs=30,pixel_format='gray16le',codec='ffv1',
               slices=24,slicecrc=1,frame_size=None,get_cmd=False):
    """Write frames to avi file using the ffv1 lossless encoder
    """

    # we probably want to include a warning about multiples of 32 for videos
    # (then we can use pyav and some speedier tools)

    if not frame_size and type(frames) is np.ndarray:
        frame_size='{0:d}x{1:d}'.format(frames.shape[2],frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size='{0:d}x{1:d}'.format(frames[0],frames[1])

    command= [ 'ffmpeg',
          '-y',
          '-threads',str(threads),
          '-framerate',str(camera_fs),
          '-f','rawvideo',
          '-s',frame_size,
          '-pix_fmt',pixel_format,
          '-i','-',
          '-an',
          '-vcodec',codec,
          '-slices',str(slices),
          '-slicecrc',str(slicecrc),
          '-r',str(camera_fs),
          filename ]

    if get_cmd:
        return command

    pipe=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    for i in range(frames.shape[0]):
        pipe.stdin.write(frames[i,...].astype('uint16').tostring())

    pipe.stdin.close()
    pipe.wait()

def read_frames(filename,frames=np.empty((0,)),threads=6,camera_fs=30,pixel_format='gray16le',frame_size=None,
              slices=24,slicecrc=1,get_cmd=False):
    """Reads in frames from the .nut/.avi file using a pipe from ffmpeg.
    Args:
        filename (str): filename to get frames from
        frames (list or 1d numpy array): list of frames to grab
        threads (int): number of threads to use for decode
        camera_fs (int): frame rate of camera in Hz
        pixel_format (str): ffmpeg pixel format of data
        frame_size (str): wxh frame size in pixels
        slices (int): number of slices to use for decode
        slicecrc (int): check integrity of slices

    Returns:
        3d numpy array:  frames x h x w
    """

    finfo=get_video_info(filename)

    if frames.size==0:
        finfo=get_video_info(filename)
        frames=np.arange(finfo['duration']).astype('int16')

    if not frame_size:
        frame_size=(finfo['width'],finfo['height'])

    command=[
        'ffmpeg',
        '-loglevel','fatal',
        '-i',filename,
        '-ss',str(datetime.timedelta(seconds=frames[0]/camera_fs)),
        '-vframes',str(len(frames)),
        '-f','image2pipe',
        '-s','{:d}x{:d}'.format(frame_size[0],frame_size[1]),
        '-pix_fmt',pixel_format,
        '-threads',str(threads),
        '-slices',str(slices),
        '-slicecrc',str(slicecrc),
        '-vcodec','rawvideo',
        '-'
    ]

    if get_cmd:
        return command

    pipe=subprocess.Popen(command,stderr=subprocess.PIPE,stdout=subprocess.PIPE)
    out, err = pipe.communicate()
    if(err): print('error',err); return None;
    video=np.frombuffer(out,dtype='uint16').reshape((len(frames),frame_size[1],frame_size[0]))
    return video

# simple command to pipe frames to an ffv1 file
def write_frames_preview(filename,frames=np.empty((0,)),threads=6,camera_fs=30,pixel_format='rgb24',codec='h264',
               slices=24,slicecrc=1,frame_size=None,depth_min=0,depth_max=80,get_cmd=False,cmap='jet'):
    """Writes out a false-colored mp4 video
    """
    if not frame_size and type(frames) is np.ndarray:
        frame_size='{0:d}x{1:d}'.format(frames.shape[2],frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size='{0:d}x{1:d}'.format(frames[0],frames[1])

    command= [ 'ffmpeg',
          '-y',
          '-threads',str(threads),
          '-framerate',str(camera_fs),
          '-f','rawvideo',
          '-s',frame_size,
          '-pix_fmt',pixel_format,
          '-i','-',
          '-an',
          '-vcodec',codec,
          '-slices',str(slices),
          '-slicecrc',str(slicecrc),
          '-r',str(camera_fs),
          '-pix_fmt','yuv420p',
          filename ]

    if get_cmd:
        return command

    pipe=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)

    # scale frames d00d

    use_cmap=plt.get_cmap(cmap)

    for i in tqdm.tqdm(range(frames.shape[0])):
        disp_img=deepcopy(frames[i,...].astype('float32'))
        disp_img=(disp_img-depth_min)/(depth_max-depth_min)
        disp_img[disp_img<0]=0
        disp_img[disp_img>1]=1
        disp_img=np.delete(use_cmap(disp_img),3,2)*255
        pipe.stdin.write(disp_img.astype('uint8').tostring())

    pipe.stdin.close()
    pipe.wait()
