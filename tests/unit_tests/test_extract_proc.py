import os
import re
import cv2
import glob
import numpy as np
import pytest
import numpy.testing as npt
from unittest import TestCase
from moseq2_extract.io.image import read_image
from moseq2_extract.extract.proc import (
    get_roi,
    crop_and_rotate_frames,
    model_smoother,
    get_frame_features,
    compute_scalars,
    clean_frames,
    get_largest_cc,
    feature_hampel_filter,
    _area_px_to_mm2,
)


class TestExtractProc(TestCase):

    def test_get_roi(self):
        # load in a bunch of ROIs where we have some ground truth
        bground_list = glob.glob("data/tiffs/bground*.tiff")

        for bground in bground_list:
            tmp = read_image(bground, scale=True)

            if re.search(r"gradient", bground) is not None:
                roi = get_roi(
                    tmp.astype("float32"),
                    depth_range=(750, 900),
                    iters=5000,
                    noise_tolerance=30,
                    bg_roi_gradient_filter=True,
                )
            else:
                roi = get_roi(
                    tmp.astype("float32"),
                    depth_range=(650, 750),
                    iters=5000,
                    noise_tolerance=30,
                )

            fname = os.path.basename(bground)
            dirname = os.path.dirname(bground)
            roi_file = "roi{}_01.tiff".format(re.search(r"\_[a-z|A-Z]*", fname).group())

            ground_truth = read_image(os.path.join(dirname, roi_file), scale=True)

            frac_nonoverlap_roi1 = np.empty((2,))
            frac_nonoverlap_roi2 = np.empty((2,))

            frac_nonoverlap_roi1[0] = np.mean(np.logical_xor(ground_truth, roi[0][0]))

            roi_file2 = "roi{}_02.tiff".format(
                re.search(r"\_[a-z|A-Z]*", fname).group()
            )

            if os.path.exists(os.path.join(dirname, roi_file2)):
                ground_truth = read_image(os.path.join(dirname, roi_file2), scale=True)
                frac_nonoverlap_roi2[0] = np.mean(
                    np.logical_xor(ground_truth, roi[0][1])
                )
                frac_nonoverlap_roi2[1] = np.mean(
                    np.logical_xor(ground_truth, roi[0][0])
                )
                frac_nonoverlap_roi1[1] = np.mean(
                    np.logical_xor(ground_truth, roi[0][1])
                )

                assert np.min(frac_nonoverlap_roi2) < 0.2

            assert np.min(frac_nonoverlap_roi1) < 0.2

    def test_crop_and_rotate(self):

        fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))

        tmp_image = np.zeros((80, 80), dtype="int8")
        center = np.array(tmp_image.shape) // 2

        mouse_dims = np.array(fake_mouse.shape) // 2

        tmp_image[
            center[0] - mouse_dims[0] : center[0] + mouse_dims[0],
            center[1] - mouse_dims[1] : center[1] + mouse_dims[1],
        ] = fake_mouse

        rotations = np.random.rand(100) * 180 - 90
        features = {}

        features["centroid"] = np.tile(center, (len(rotations), 1))
        features["orientation"] = np.deg2rad(rotations)

        fake_movie = np.zeros(
            (len(rotations), tmp_image.shape[0], tmp_image.shape[1]), dtype="float32"
        )

        for i, rotation in enumerate(rotations):

            rot_mat = cv2.getRotationMatrix2D(tuple(center), rotation, 1)
            fake_movie[i] = cv2.warpAffine(
                tmp_image.astype("float32"), rot_mat, (80, 80)
            )

        cropped_and_rotated = (
            crop_and_rotate_frames(fake_movie, features=features) > 0.4
        ).astype(tmp_image.dtype)
        percent_pixels_diff = np.mean(np.abs(cropped_and_rotated - tmp_image)) * 1e2

        assert percent_pixels_diff < 0.1

    def test_get_frame_features(self):

        fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))

        tmp_image = np.zeros((80, 80), dtype="int8")
        center = np.array(tmp_image.shape) // 2

        mouse_dims = np.array(fake_mouse.shape) // 2

        tmp_image[
            center[0] - mouse_dims[0] : center[0] + mouse_dims[0],
            center[1] - mouse_dims[1] : center[1] + mouse_dims[1],
        ] = fake_mouse

        fake_movie = np.tile(tmp_image, (100, 1, 1))
        fake_features, mask = get_frame_features(fake_movie, frame_threshold=0.01)

        npt.assert_almost_equal(fake_features["orientation"], 0, 2)
        npt.assert_almost_equal(
            fake_features["centroid"], np.tile(center, (fake_movie.shape[0], 1)), 0.1
        )
        npt.assert_array_almost_equal(
            fake_features["axis_length"],
            np.tile([30, 20], (fake_movie.shape[0], 1)),
            0.1,
        )

    def test_compute_scalars(self):

        fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))

        tmp_image = np.zeros((80, 80), dtype="int8")
        center = np.array(tmp_image.shape) // 2

        mouse_dims = np.array(fake_mouse.shape) // 2

        tmp_image[
            center[0] - mouse_dims[0] : center[0] + mouse_dims[0],
            center[1] - mouse_dims[1] : center[1] + mouse_dims[1],
        ] = fake_mouse

        fake_movie = np.tile(tmp_image, (100, 1, 1))
        fake_features, mask = get_frame_features(fake_movie, frame_threshold=0.01)
        fake_scalars = compute_scalars(fake_movie, fake_features, min_height=0.01)

        npt.assert_almost_equal(fake_scalars["centroid_x_px"], center[1], 0.1)
        npt.assert_almost_equal(fake_scalars["centroid_y_px"], center[0], 0.1)
        npt.assert_almost_equal(fake_scalars["angle"], 0, 2)
        npt.assert_almost_equal(fake_scalars["width_px"], mouse_dims[0] * 2, 0.1)
        npt.assert_almost_equal(fake_scalars["length_px"], mouse_dims[1] * 2, 0.1)
        npt.assert_almost_equal(fake_scalars["height_ave_mm"], 1, 1)
        npt.assert_almost_equal(fake_scalars["velocity_2d_px"], 0, 1)
        npt.assert_almost_equal(fake_scalars["velocity_3d_px"], 0, 1)
        npt.assert_almost_equal(fake_scalars["velocity_theta"], 0, 1)
        npt.assert_almost_equal(fake_scalars["area_px"], np.sum(tmp_image), 0.1)

    def test_compute_scalars_area_mm_scales_quadratically_with_depth(self):
        # convert_pxs_to_mm is linear in true_depth, so doubling true_depth
        # doubles both per-axis mm/px factors. An area therefore has to scale by
        # 4, while any single linear factor would only scale it by 2.
        #
        # This is a metamorphic test of the scaling law: pixel inputs are held
        # fixed while only true_depth varies. It is not a claim that the two
        # depths represent physically consistent recordings.
        frames = np.zeros((1, 424, 512), dtype="float32")
        frames[0, 88:90, 136:138] = 20.0

        # Float centroid on purpose: an integer centroid array is truncated by
        # the zeros_like call in convert_pxs_to_mm, which is a separate issue.
        track_features = {
            "centroid": np.array([[137.0, 89.0]]),
            "orientation": np.array([0.0]),
            "axis_length": np.array([[4.0, 2.0]]),
        }

        area_at_d = compute_scalars(
            frames, track_features, min_height=10, max_height=100,
            true_depth=500.0,
        )["area_mm"][0]
        area_at_2d = compute_scalars(
            frames, track_features, min_height=10, max_height=100,
            true_depth=1000.0,
        )["area_mm"][0]

        assert area_at_d > 0
        assert area_at_2d / area_at_d == pytest.approx(4.0, rel=1e-6)

    def test_area_px_to_mm2_scales_with_each_axis_independently(self):
        # The depth test above cannot distinguish the product of both axis
        # factors from the square of their mean, since true_depth scales both
        # factors together. Scaling one axis at a time pins the product.
        area_px = np.array([4.0])
        px_to_mm = np.array([[0.5, 0.25]])

        base = _area_px_to_mm2(area_px, px_to_mm)[0]
        assert base == pytest.approx(4.0 * 0.5 * 0.25, rel=1e-12)

        x_scaled = _area_px_to_mm2(area_px, px_to_mm * np.array([[3.0, 1.0]]))[0]
        y_scaled = _area_px_to_mm2(area_px, px_to_mm * np.array([[1.0, 3.0]]))[0]

        assert x_scaled / base == pytest.approx(3.0, rel=1e-12)
        assert y_scaled / base == pytest.approx(3.0, rel=1e-12)

    def test_get_largest_cc(self):

        fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))

        tmp_image = np.zeros((80, 80), dtype="int8")
        center = np.array(tmp_image.shape) // 2

        mouse_dims = np.array(fake_mouse.shape) // 2

        tmp_image[
            center[0] - mouse_dims[0] : center[0] + mouse_dims[0],
            center[1] - mouse_dims[1] : center[1] + mouse_dims[1],
        ] = fake_mouse

        fake_movie = np.tile(tmp_image, (100, 1, 1))
        largest_cc_movie = get_largest_cc(fake_movie)

        npt.assert_array_almost_equal(fake_movie, largest_cc_movie, 3)

    def test_clean_frames(self):

        edge_size = 40
        points = np.arange(-edge_size, edge_size)
        sig1 = 10
        sig2 = 20

        kernel = np.exp(-(points**2.0) / (2.0 * sig1**2.0))
        kernel2 = np.exp(-(points**2.0) / (2.0 * sig2**2.0))

        kernel_full = np.outer(kernel, kernel2)
        kernel_full /= np.max(kernel_full)
        kernel_full *= 50

        fake_mouse = kernel_full
        fake_mouse[fake_mouse < 5] = 0

        fake_movie = np.tile(fake_mouse, (100, 1, 1))
        cleaned_fake_movie = clean_frames(fake_movie, prefilter_time=(3,))

    def test_feature_hampel_filter(self):

        fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))

        tmp_image = np.zeros((80, 80), dtype="int8")
        center = np.array(tmp_image.shape) // 2

        mouse_dims = np.array(fake_mouse.shape) // 2

        tmp_image[
            center[0] - mouse_dims[0] : center[0] + mouse_dims[0],
            center[1] - mouse_dims[1] : center[1] + mouse_dims[1],
        ] = fake_mouse

        fake_movie = np.tile(tmp_image, (100, 1, 1))
        fake_features, mask = get_frame_features(fake_movie, frame_threshold=0.01)

        smoothed_features = feature_hampel_filter(
            fake_features,
            centroid_hampel_span=5,
            centroid_hampel_sig=5,
            angle_hampel_span=5,
            angle_hampel_sig=3,
        )

        assert list(smoothed_features.keys()) == list(fake_features.keys())
        for k in smoothed_features.keys():
            assert smoothed_features[k].all() == fake_features[k].all()

    def test_model_smoother(self):

        fake_mouse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))

        tmp_image = np.zeros((80, 80), dtype="int8")
        center = np.array(tmp_image.shape) // 2

        mouse_dims = np.array(fake_mouse.shape) // 2

        tmp_image[
            center[0] - mouse_dims[0] : center[0] + mouse_dims[0],
            center[1] - mouse_dims[1] : center[1] + mouse_dims[1],
        ] = fake_mouse

        fake_movie = np.tile(tmp_image, (100, 1, 1))
        fake_features, mask = get_frame_features(fake_movie, frame_threshold=0.01)

        smoothed_feats = model_smoother(fake_features, ll=None, clips=(-300, -125))
        assert smoothed_feats == fake_features

        test_ll = np.random.randint(-25, 256, size=(100, 30, 20), dtype="int16")
        fake_features["axis_length"][0] = [np.nan, np.nan]

        smoothed_feats = model_smoother(fake_features, ll=test_ll, clips=(-300, -125))
        assert smoothed_feats == fake_features

        test_ll = np.random.randint(-25, 256, size=(100, 30, 20), dtype="int16")
        fake_features["axis_length"] = np.array(
            [x[0] for x in fake_features["axis_length"]]
        )
        print(fake_features["axis_length"])
        fake_features["axis_length"][0] = np.nan

        smoothed_feats = model_smoother(fake_features, ll=test_ll, clips=(-300, -125))
        assert smoothed_feats == fake_features
