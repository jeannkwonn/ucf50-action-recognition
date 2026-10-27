"""Shared pytest fixtures: small fake frame folders and videos built in a temporary directory."""

import cv2
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def make_clip():
    """Return a function that writes a folder of tiny JPEG frames and returns its path."""
    def _make_clip(root, class_name, clip_name, n_frames=4, size=32):
        clip_dir = root / class_name / clip_name
        clip_dir.mkdir(parents=True)
        for i in range(n_frames):
            # Frame i is filled with the value i * 10, so the frame order can be checked later.
            pixels = np.full((size, size, 3), i * 10, dtype=np.uint8)
            Image.fromarray(pixels).save(clip_dir / f'frame_{i:03d}.jpg', quality=100)
        return clip_dir
    return _make_clip


@pytest.fixture
def make_video():
    """Return a function that writes a tiny .avi video and returns its path."""
    def _make_video(path, n_frames=30, size=32):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (size, size))
        for i in range(n_frames):
            writer.write(np.full((size, size, 3), (i * 8) % 256, dtype=np.uint8))
        writer.release()
        return path
    return _make_video
