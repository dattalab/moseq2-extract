import numpy as np
import numpy.testing as npt
from unittest import TestCase
from tempfile import TemporaryDirectory, NamedTemporaryFile
from moseq2_extract.io.video import read_frames_raw, get_raw_info,\
    read_frames, write_frames, get_video_info, write_frames_preview,\
    get_movie_info, load_movie_data


class TestVideoIO(TestCase):
    def test_read_frames_raw(self):

        with TemporaryDirectory() as tmp:
            data_path = NamedTemporaryFile(prefix=tmp, suffix=".dat")

        test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')
        test_data.tofile(data_path.name)

        read_data = read_frames_raw(data_path.name)
        npt.assert_array_equal(test_data, read_data)


    def test_get_raw_info(self):

        with TemporaryDirectory() as tmp:
            data_path = NamedTemporaryFile(prefix=tmp, suffix=".dat")


            test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')
            test_data.tofile(data_path.name)

            vid_info = get_raw_info(data_path.name)

            npt.assert_equal(vid_info['bytes'], test_data.nbytes)
            npt.assert_equal(vid_info['nframes'], test_data.shape[0])
            npt.assert_equal(vid_info['dims'], (test_data.shape[2], test_data.shape[1]))
            npt.assert_equal(vid_info['bytes_per_frame'], test_data.nbytes / test_data.shape[0])


    def test_ffv1(self):

        with TemporaryDirectory() as tmp:
            data_path = NamedTemporaryFile(prefix=tmp, suffix=".avi")

            test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')
            test_data.tofile(data_path.name)

            write_frames(data_path.name, test_data, fps=30)
            read_data = read_frames(data_path.name)

            vid_info = get_video_info(data_path.name)

            npt.assert_equal(test_data, read_data)
            npt.assert_equal(vid_info['fps'], 30)
            npt.assert_equal((vid_info['dims']), (test_data.shape[2], test_data.shape[1]))
            npt.assert_equal(vid_info['nframes'], test_data.shape[0])


    def test_write_frames_preview(self):

        with TemporaryDirectory() as tmp:
            data_path = NamedTemporaryFile(prefix=tmp, suffix=".avi")

            test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')
            write_frames_preview(data_path.name, test_data, fps=30)


    def test_get_movie_info(self):
        with TemporaryDirectory() as tmp:
            for suff in ['.avi', '.mkv']:
                avi_path = NamedTemporaryFile(prefix=tmp, suffix=suff)
                dat_path = NamedTemporaryFile(prefix=tmp, suffix=".dat")

                test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')

                write_frames(avi_path.name, test_data, fps=30)
                test_data.tofile(dat_path.name)

                vid_info = get_movie_info(avi_path.name)
                raw_info = get_movie_info(dat_path.name)

                npt.assert_equal(vid_info['dims'], raw_info['dims'])
                npt.assert_equal(vid_info['nframes'], raw_info['nframes'])


    def test_load_movie_data(self):
        with TemporaryDirectory() as tmp:
            avi_path = NamedTemporaryFile(prefix=tmp, suffix=".avi")
            dat_path = NamedTemporaryFile(prefix=tmp, suffix=".dat")

            test_data = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')

            write_frames(avi_path.name, test_data, fps=30)
            test_data.tofile(dat_path.name)

            read_data_vid = load_movie_data(avi_path.name)
            read_data_raw = load_movie_data(dat_path.name)

            npt.assert_array_equal(read_data_vid, read_data_raw)
