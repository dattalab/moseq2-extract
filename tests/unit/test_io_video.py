import numpy.testing as npt
import numpy as np
import pytest
import subprocess
import datetime
import os

from moseq2_extract.io.video import read_frames_raw, get_raw_info,\
    read_frames, write_frames, get_video_info, write_frames_preview,\
    get_movie_info, load_movie_data


@pytest.fixture(scope='function')
def video_file(tmpdir):
    f = tmpdir.mkdir('test_dir')
    return str(f)

def test_get_video_info():
    # original param: filename
    filename = 'tests/test_video_files/test_vid.avi'

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
    ffmpeg.wait()
    print(ffmpeg.returncode)
    if (err):
        print(err)

    out = out.decode().split('\n')

    mock = {'file': filename,
     'dims': (int(out[0]), int(out[1])),
     'fps': float(out[2].split('/')[0]) / float(out[2].split('/')[1]),
     'nframes': int(out[3])}

    for k,v in mock.items():
        if k == None:
            pytest.fail('missing key')
        if v == None:
            pytest.fail('missing val')

def test_read_frames():
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
    filename = 'tests/test_video_files/test_vid.avi'
    threads = 6
    fps = 30
    slices = 24
    slicecrc = 1
    pixel_format = 'gray16le'

    mock_info = get_video_info(filename)
    if mock_info == None:
        pytest.fail('info is None')

    for k,v in mock_info.items():
        if k == None or not k.isalpha():
            print(k)
            pytest.fail('info key is incorrect')
        if v == None:
            print(v)
            pytest.fail('info value is incorrect')

    frame_stride = 500 # default
    frame_idx = np.arange(0, mock_info['nframes'], frame_stride)
    frames = np.arange(mock_info['nframes']).astype('int16')
    frame_size = mock_info['dims']

    command = [
        'ffmpeg',
        '-loglevel', 'fatal',
        '-ss', str(datetime.timedelta(seconds=frames[0] / fps)),
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

    for i, frame in enumerate(frame_idx):
        pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = pipe.communicate()
        if (err):
            print('error', err)
            pytest.fail('Popen error')
            return None
        video = np.frombuffer(out, dtype='uint16').reshape((len(frames), frame_size[1], frame_size[0]))
        if video.shape < (1,):
            pytest.fail('video is incorrect shape')

def test_read_frames_raw(video_file):

    test_path = os.path.join(video_file, 'test_data.data')
    test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')
    test_data.tofile(test_path)

    read_data = read_frames_raw(test_path)
    npt.assert_array_equal(test_data, read_data)


def test_get_raw_info(video_file):

    test_path = os.path.join(video_file, 'test_data.data')
    test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')
    test_data.tofile(test_path)

    vid_info = get_raw_info(test_path)

    npt.assert_equal(vid_info['bytes'], test_data.nbytes)
    npt.assert_equal(vid_info['nframes'], test_data.shape[0])
    npt.assert_equal(vid_info['dims'], (test_data.shape[2], test_data.shape[1]))
    npt.assert_equal(vid_info['bytes_per_frame'], test_data.nbytes / test_data.shape[0])


def test_ffv1(video_file):

    test_path = os.path.join(video_file, 'test_data.avi')
    test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')

    write_frames(test_path, test_data, fps=30)
    read_data = read_frames(test_path)

    vid_info = get_video_info(test_path)

    npt.assert_equal(test_data, read_data)
    npt.assert_equal(vid_info['fps'], 30)
    npt.assert_equal((vid_info['dims']), (test_data.shape[2], test_data.shape[1]))
    npt.assert_equal(vid_info['nframes'], test_data.shape[0])


def test_write_frames_preview(video_file):

    test_path = os.path.join(video_file, 'test_data.avi')
    test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')

    write_frames_preview(test_path, test_data, fps=30)


def test_get_movie_info(video_file):

    test_path_vid = os.path.join(video_file, 'test_data.avi')
    test_path_raw = os.path.join(video_file, 'test_data.dat')
    test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')

    write_frames(test_path_vid, test_data, fps=30)
    test_data.tofile(test_path_raw)

    vid_info = get_movie_info(test_path_vid)
    raw_info = get_movie_info(test_path_raw)

    npt.assert_equal(vid_info['dims'], raw_info['dims'])
    npt.assert_equal(vid_info['nframes'], raw_info['nframes'])


def test_load_movie_data(video_file):

    test_path_vid = os.path.join(video_file, 'test_data.avi')
    test_path_raw = os.path.join(video_file, 'test_data.dat')
    test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')

    write_frames(test_path_vid, test_data, fps=30)
    test_data.tofile(test_path_raw)

    read_data_vid = load_movie_data(test_path_vid)
    read_data_raw = load_movie_data(test_path_raw)

    npt.assert_array_equal(read_data_vid, read_data_raw)
