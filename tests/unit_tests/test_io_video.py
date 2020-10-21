import os
import math
import tarfile
import numpy as np
from os.path import join
import numpy.testing as npt
from unittest import TestCase
from ..integration_tests.test_cli import write_fake_movie
from moseq2_extract.io.video import read_frames_raw, get_raw_info,\
    read_frames, write_frames, get_video_info, write_frames_preview,\
    get_movie_info, load_movie_data

class TestVideoIO(TestCase):
    def test_read_frames_raw(self):

        data_path = 'data/fake_depth.dat'

        test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')
        test_data.tofile(data_path)

        read_data = read_frames_raw(data_path)
        npt.assert_array_equal(test_data, read_data)
        os.remove(data_path)

    def test_get_raw_info(self):

        data_path = 'data/fake_raw_depth.dat'

        test_data = np.random.randint(0, 256, size=(100, 424, 512), dtype='int16')
        test_data.tofile(data_path)

        vid_info = get_raw_info(data_path)

        npt.assert_equal(vid_info['bytes'], test_data.nbytes)
        npt.assert_equal(vid_info['nframes'], test_data.shape[0])
        npt.assert_equal(vid_info['dims'], (test_data.shape[2], test_data.shape[1]))
        npt.assert_equal(vid_info['bytes_per_frame'], test_data.nbytes / test_data.shape[0])
        os.remove(data_path)

        avi_path = 'data/test-out.avi'
        vid_info = get_raw_info(avi_path)

        npt.assert_equal(vid_info['bytes'], 15824724)
        npt.assert_equal(vid_info['nframes'], test_data.shape[0])
        npt.assert_equal(vid_info['dims'], (test_data.shape[2], test_data.shape[1]))
        npt.assert_equal(vid_info['bytes_per_frame'], test_data.nbytes / test_data.shape[0])

    def test_ffv1(self):

        data_path = 'data/fake_ffv1_depth.avi'

        test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')
        test_data.tofile(data_path)

        write_frames(data_path, test_data, fps=30)
        read_data = read_frames(data_path)

        vid_info = get_video_info(data_path)

        npt.assert_equal(test_data, read_data)
        npt.assert_equal(vid_info['fps'], 30)
        npt.assert_equal((vid_info['dims']), (test_data.shape[2], test_data.shape[1]))
        npt.assert_equal(vid_info['nframes'], test_data.shape[0])
        os.remove(data_path)

    def test_write_frames_preview(self):

        data_path = 'data/fake_preview_depth.avi'

        test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')
        write_frames_preview(data_path, test_data, fps=30, frame_range=range(len(test_data)))
        os.remove(data_path)

    def test_get_movie_info(self):

        avi_path = 'data/fake_movie_info.avi'
        dat_path = 'data/fake_movie_info.dat'

        test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')

        write_frames(avi_path, test_data, fps=30)
        test_data.tofile(dat_path)

        vid_info = get_movie_info(avi_path)
        raw_info = get_movie_info(dat_path)

        npt.assert_equal(vid_info['dims'], raw_info['dims'])
        npt.assert_equal(vid_info['nframes'], raw_info['nframes'])
        os.remove(avi_path)
        os.remove(dat_path)

    def test_load_movie_data(self):
        avi_path = 'data/fake_movie_data.avi'
        dat_path = 'data/fake_movie_data.dat'

        test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')

        write_frames(avi_path, test_data, fps=30)
        test_data.tofile(dat_path)

        read_data_vid = load_movie_data(avi_path)
        read_data_raw = load_movie_data(dat_path)

        npt.assert_array_equal(read_data_vid, read_data_raw)
        os.remove(avi_path)
        os.remove(dat_path)
