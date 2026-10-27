"""Unit tests for utils.py: frame extraction, saving frames, and image transforms."""

import os

import pytest
import torch

from utils import (compose_data_transforms, get_device, get_frame_pairs, get_frames,
                   store_frames, transform_stats)


def test_get_frames_returns_exactly_the_requested_number(tmp_path, make_video):
    """Asking for 16 frames gives 16 frames (the old code returned 17)."""
    video = make_video(tmp_path / 'clip.avi', n_frames=30)
    frames, video_length = get_frames(str(video), n_frames=16)
    assert len(frames) == 16
    assert video_length == 30
    assert frames[0].shape == (32, 32, 3)


def test_get_frames_handles_missing_video(tmp_path):
    """A video that cannot be opened gives an empty result instead of crashing."""
    frames, video_length = get_frames(str(tmp_path / 'missing.avi'), n_frames=4)
    assert not frames
    assert video_length == 0


def test_store_frames_uses_zero_padded_names(tmp_path, make_video):
    """Zero padded names mean alphabetical order is also time order."""
    video = make_video(tmp_path / 'clip.avi', n_frames=12)
    frames, _ = get_frames(str(video), n_frames=12)
    out_dir = tmp_path / 'frames'
    store_frames(frames, str(out_dir))
    names = sorted(os.listdir(out_dir))
    assert names[0] == 'frame_000.jpg'
    assert names[-1] == 'frame_011.jpg'
    assert len(names) == 12


def test_transform_stats_for_lrcn_and_unknown_model():
    """The LRCN uses 224 x 224 images, and an unknown model name is an error."""
    height, width, mean, std = transform_stats('lrcn')
    assert (height, width) == (224, 224)
    assert len(mean) == len(std) == 3
    with pytest.raises(ValueError):
        transform_stats('unknown')


@pytest.mark.parametrize('strong_aug', [False, True])
def test_transforms_output_shape(strong_aug):
    """Train and validation transforms both produce (T, 3, 224, 224) clips."""
    height, width, mean, std = transform_stats('lrcn')
    train_tf, val_tf = compose_data_transforms(height, width, mean, std, strong_aug)
    clip = torch.rand(4, 3, 60, 80)
    assert train_tf(clip).shape == (4, 3, 224, 224)
    assert val_tf(clip).shape == (4, 3, 224, 224)


@pytest.mark.parametrize('strong_aug', [False, True])
def test_augmentation_is_the_same_for_every_frame_of_a_clip(strong_aug):
    """If every input frame is identical, every output frame must be identical too."""
    height, width, mean, std = transform_stats('lrcn')
    train_tf, _ = compose_data_transforms(height, width, mean, std, strong_aug)
    frame = torch.rand(1, 3, 60, 80)
    clip = frame.repeat(6, 1, 1, 1)
    for seed in range(5):
        torch.manual_seed(seed)
        out = train_tf(clip)
        assert torch.allclose(out, out[:1].expand_as(out))


def test_get_device_returns_a_torch_device():
    """get_device picks one of the supported devices."""
    assert get_device().type in ('cuda', 'mps', 'cpu')


def test_get_frame_pairs_returns_two_frames_per_pair(tmp_path, make_video):
    """Asking for 5 pairs gives 10 frames, ordered first, second, first, second, ..."""
    video = make_video(tmp_path / 'clip.avi', n_frames=40)
    frames, video_length = get_frame_pairs(str(video), n_pairs=5, gap=4)
    assert len(frames) == 10
    assert video_length == 40
    # The test video gets brighter over time, so the second frame of a pair is brighter.
    for first, second in zip(frames[0::2], frames[1::2]):
        assert second.mean() > first.mean()


def test_get_frame_pairs_rejects_a_video_that_is_too_short(tmp_path, make_video):
    """A video shorter than the gap cannot make a pair."""
    video = make_video(tmp_path / 'short.avi', n_frames=3)
    frames, _ = get_frame_pairs(str(video), n_pairs=2, gap=4)
    assert not frames
