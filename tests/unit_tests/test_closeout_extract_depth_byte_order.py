"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest


@pytest.mark.parametrize("order", ["be", "le"])
def test_video_depth_decoding_respects_declared_byte_order(monkeypatch, order):
    from types import SimpleNamespace
    from moseq2_extract.io import video
    samples = np.array([673, 1024], dtype=(">u2" if order == "be" else "<u2"))
    monkeypatch.setattr(video.subprocess, "Popen", lambda *a, **kw:
                        SimpleNamespace(communicate=lambda: (samples.tobytes(), b"")))
    actual = video.read_frames("synthetic.mkv", frames=[0], pixel_format="gray16" + order,
                               finfo={"nframes": 1, "dims": (2, 1)}, mapping=0)
    np.testing.assert_array_equal(actual, [[[673, 1024]]])
