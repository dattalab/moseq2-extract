import cv2
import h5py
import json
import numpy as np
import numpy.testing as npt
from unittest import TestCase
from moseq2_extract.cli import find_roi
from tempfile import TemporaryDirectory, NamedTemporaryFile
from moseq2_extract.util import gen_batch_sequence, load_metadata, load_timestamps,\
    select_strel, scalar_attributes, save_dict_contents_to_h5, click_param_annot

class testExtractUtils(TestCase):
    def test_gen_batch_sequence(self):

        tmp_list = [range(0, 10),
                    range(5, 15),
                    range(10, 20),
                    range(15, 25)]

        gen_list = list(gen_batch_sequence(25, 10, 5))

        assert(gen_list == tmp_list)


    def test_load_timestamps(self):

        with TemporaryDirectory() as tmp:
            txt_path = NamedTemporaryFile(prefix=tmp, suffix=".txt")

            tmp_timestamps = np.arange(0, 5, .05)
            with open(txt_path.name, 'w') as f:
                for timestamp in tmp_timestamps:
                    print('{}'.format(str(timestamp)), file=f)

            loaded_timestamps = load_timestamps(txt_path.name)
            npt.assert_almost_equal(loaded_timestamps, tmp_timestamps, 10)


    def test_load_metadata(self):

        tmp_dict = {
            'test': 'test2'
        }

        with TemporaryDirectory() as tmp:
            json_file = NamedTemporaryFile(prefix=tmp, suffix=".json")
            with open(json_file.name, 'w') as f:
                json.dump(tmp_dict, f)

            loaded_dict = load_metadata(json_file.name)

            assert(loaded_dict == tmp_dict)


    def test_select_strel(self):

        strel = select_strel('ellipse', size=(9, 9))
        npt.assert_equal(strel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

        strel = select_strel('rectangle', size=(9, 9))
        npt.assert_equal(strel, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))

        strel = select_strel('sdfdfsf', size=(9, 9))
        npt.assert_equal(strel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


    def test_scalar_attributes(self):

        dct = scalar_attributes()

        assert(dct is not None)


    def test_save_dict_contents_to_h5(self):

        tmp_dic = {
            'subdict': {
                'sd_tuple': (0,1),
                'sd_string': 'quick brown fox',
                'sd_integer': 1,
                'sd_float': 1.0,
                'sd_bool': False,
                'sd_list': [1,2,3],
            },
            'tuple': (0,1),
            'string': 'quick brown fox',
            'integer': 1,
            'float': 1.0,
            'bool': False,
            'list': [1,2,3],
        }
        with TemporaryDirectory() as tmp:
            fpath = NamedTemporaryFile(prefix=tmp, suffix=".h5")
            f = h5py.File(fpath.name, 'w')
            save_dict_contents_to_h5(f, tmp_dic, tmp)
            f.close()

            def h5_to_dict(h5file, path):
                ans = {}
                if not path.endswith('/'):
                    path = path + '/'
                for key, item in h5file[path].items():
                    if type(item) is h5py.Dataset:
                        ans[key] = item[()]
                    elif type(item) is h5py.Group:
                        ans[key] = h5_to_dict(h5file, path + key + '/')
                return ans

            result = h5_to_dict(h5py.File(fpath.name, 'r'), tmp)
            npt.assert_equal(result, tmp_dic)

    def test_click_param_annot(self):
        ref_dict = {
            'bg_roi_dilate': 'Size of strel to dilate roi',
            'bg_roi_shape': 'Shape to use to dilate roi (ellipse or rect)',
            'bg_roi_index': 'Index of roi to use',
            'bg_roi_weights': 'ROI feature weighting (area, extent, dist)',
            'bg_roi_depth_range': 'Range to search for floor of arena (in mm)',
            'bg_roi_gradient_filter': 'Exclude walls with gradient filtering',
            'bg_roi_gradient_threshold': 'Gradient must be < this to include points',
            'bg_roi_gradient_kernel': 'Kernel size for Sobel gradient filtering',
            'bg_sort_roi_by_position': 'Sort ROIs by position',
            'bg_sort_roi_by_position_max_rois': 'Max original ROIs to sort by position',
            'dilate_iterations': 'Number of dilation iterations to increase bucket floor size.',
            'bg_roi_fill_holes': 'Fill holes in ROI',
            'output_dir': 'Output directory',
            'use_plane_bground': 'Use plane fit for background',
            'config_file': None
        }

        test_dict = click_param_annot(find_roi)
        npt.assert_equal(ref_dict, test_dict)