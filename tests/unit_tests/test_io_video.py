import numpy.testing as npt
import numpy as np
import pytest
import os
from moseq2_extract.io.video import read_frames_raw, get_raw_info,\
    read_frames, write_frames, get_video_info, write_frames_preview,\
    get_movie_info, load_movie_data


@pytest.fixture(scope='function')
def video_file(tmpdir):
    f = tmpdir.mkdir('test_dir')
    return str(f)


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
