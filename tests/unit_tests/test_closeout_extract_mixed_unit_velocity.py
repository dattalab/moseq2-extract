"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pytest


def test_mixed_unit_velocity_is_not_emitted():
    from moseq2_extract.extract.proc import compute_scalars
    from moseq2_extract.util import scalar_attributes
    frames = np.stack([np.full((8, 8), h, dtype=float) for h in [20, 32, 44]])
    track = {"centroid": np.array([[256., 212.], [259., 216.], [262., 220.]]),
             "orientation": np.zeros(3), "axis_length": np.tile([4., 2.], (3, 1))}
    actual = compute_scalars(frames, track)
    assert "velocity_3d_px" not in actual
    assert "velocity_3d_px" not in scalar_attributes()
    np.testing.assert_allclose(actual["velocity_2d_px"], [0, 5, 5])
    np.testing.assert_allclose(actual["height_ave_mm"], [20, 32, 44])
    np.testing.assert_allclose(actual["velocity_3d_mm"][1:] ** 2,
                               actual["velocity_2d_mm"][1:] ** 2 + 144)
