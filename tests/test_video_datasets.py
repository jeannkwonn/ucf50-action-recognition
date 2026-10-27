"""Unit tests for video_datasets.py: group parsing, sorting, group-based splitting and loading."""

import pytest
import torch

from video_datasets import (VideoDataset, collate_fn_rnn, dataset_split, get_group,
                            load_dataset, natural_sort_key)


def test_natural_sort_key_orders_numbers_as_numbers():
    """frame2 must come before frame10 (plain alphabetical order gets this wrong)."""
    names = ['frame10.jpg', 'frame2.jpg', 'frame1.jpg']
    assert sorted(names, key=natural_sort_key) == ['frame1.jpg', 'frame2.jpg', 'frame10.jpg']


def test_get_group_reads_the_group_number():
    """The group number is read from the "_gXX_" part of the clip name."""
    assert get_group('data/Biking/v_Biking_g07_c03') == 7
    assert get_group('data/Biking/v_Biking_g25_c01/') == 25


def test_get_group_rejects_names_without_a_group():
    """A name with no group tag is an error, not a silent guess."""
    with pytest.raises(ValueError):
        get_group('data/Biking/some_clip')


def make_fake_dataset(n_classes=3, n_groups=25, clips_per_group=2):
    """Build a dictionary like load_dataset does, without any real files."""
    dataset = {}
    for label in range(n_classes):
        for group in range(1, n_groups + 1):
            for clip in range(1, clips_per_group + 1):
                dataset[f'data/Class{label}/v_Class{label}_g{group:02d}_c{clip:02d}'] = label
    return dataset


def test_dataset_split_has_no_group_overlap():
    """The data leak fix: no group may appear in more than one split."""
    train, val, test = dataset_split(make_fake_dataset())
    train_groups = {get_group(path) for path, _ in train}
    val_groups = {get_group(path) for path, _ in val}
    test_groups = {get_group(path) for path, _ in test}
    assert train_groups == set(range(1, 18))
    assert val_groups == {18, 19, 20, 21}
    assert test_groups == {22, 23, 24, 25}


def test_dataset_split_keeps_every_clip_once():
    """Every clip lands in exactly one split."""
    dataset = make_fake_dataset()
    train, val, test = dataset_split(dataset)
    assert len(train) + len(val) + len(test) == len(dataset)
    all_paths = [path for path, _ in train + val + test]
    assert len(set(all_paths)) == len(dataset)


def test_dataset_split_rejects_overlapping_groups():
    """The same group cannot be used for both validation and testing."""
    with pytest.raises(ValueError):
        dataset_split(make_fake_dataset(), val_groups=(18, 19), test_groups=(19, 20))


def test_dataset_split_rejects_empty_split():
    """A split with no clips is an error."""
    with pytest.raises(ValueError):
        dataset_split(make_fake_dataset(n_groups=5))


def test_load_dataset_skips_files_and_labels_classes_in_order(tmp_path, make_clip):
    """Only folders count as classes (so .DS_Store is ignored) and labels follow sorted names."""
    make_clip(tmp_path, 'Biking', 'v_Biking_g01_c01')
    make_clip(tmp_path, 'Archery', 'v_Archery_g01_c01')
    (tmp_path / '.DS_Store').write_text('junk')
    dataset, label_dict = load_dataset(str(tmp_path))
    assert label_dict == {'Archery': 0, 'Biking': 1}
    assert sorted(dataset.values()) == [0, 1]


def test_video_dataset_returns_frames_in_time_order(tmp_path, make_clip):
    """Frames come back as a (T, C, H, W) tensor, ordered by frame number."""
    clip_dir = make_clip(tmp_path, 'Biking', 'v_Biking_g01_c01', n_frames=12)
    clip, label = VideoDataset([(str(clip_dir), 1)], fr_per_vid=12)[0]
    assert clip.shape == (12, 3, 32, 32)
    assert label == 1
    means = clip.mean(dim=(1, 2, 3))
    assert torch.all(means[1:] > means[:-1])


def test_video_dataset_picks_evenly_spaced_frames(tmp_path, make_clip):
    """If a clip has more frames than requested, frames are spread across the whole clip."""
    clip_dir = make_clip(tmp_path, 'Biking', 'v_Biking_g01_c01', n_frames=9)
    clip, _ = VideoDataset([(str(clip_dir), 0)], fr_per_vid=3)[0]
    assert clip.shape[0] == 3
    values = (clip.mean(dim=(1, 2, 3)) * 255).round().tolist()
    assert values == pytest.approx([0, 40, 80], abs=2)


def frame_values(clip):
    """Read back which stored frame each clip frame came from (frame i was filled with i * 10)."""
    return (clip.mean(dim=(1, 2, 3)) * 255 / 10).round().int().tolist()


def test_random_sample_picks_one_frame_per_segment_in_order(tmp_path, make_clip):
    """With temporal jitter, each of the 4 time segments gives one frame, still in time order."""
    clip_dir = make_clip(tmp_path, 'Biking', 'v_Biking_g01_c01', n_frames=16)
    dataset = VideoDataset([(str(clip_dir), 0)], fr_per_vid=4, random_sample=True)
    seen = set()
    for seed in range(20):
        torch.manual_seed(seed)
        indices = frame_values(dataset[0][0])
        seen.add(tuple(indices))
        assert len(indices) == 4
        for segment, index in enumerate(indices):
            assert segment * 4 <= index < (segment + 1) * 4
    assert len(seen) > 1


def test_random_sample_off_is_repeatable(tmp_path, make_clip):
    """Without temporal jitter the same frames are chosen every time."""
    clip_dir = make_clip(tmp_path, 'Biking', 'v_Biking_g01_c01', n_frames=16)
    dataset = VideoDataset([(str(clip_dir), 0)], fr_per_vid=4)
    assert frame_values(dataset[0][0]) == frame_values(dataset[0][0]) == [0, 5, 10, 15]


def test_frame_diff_returns_the_change_between_pair_frames(tmp_path, make_clip):
    """Stored frames i hold the value i * 10, so each pair (2k, 2k + 1) differs by 10 / 255."""
    clip_dir = make_clip(tmp_path, 'Biking', 'v_Biking_g01_c01', n_frames=8)
    clip, _ = VideoDataset([(str(clip_dir), 0)], fr_per_vid=4, frame_diff=True)[0]
    assert clip.shape == (4, 3, 32, 32)
    assert clip.mean().item() == pytest.approx(10 / 255, abs=0.01)


def test_frame_diff_keeps_only_the_requested_number_of_pairs(tmp_path, make_clip):
    """With 12 stored frames (6 pairs) and 3 requested, 3 pairs are kept."""
    clip_dir = make_clip(tmp_path, 'Biking', 'v_Biking_g01_c01', n_frames=12)
    clip, _ = VideoDataset([(str(clip_dir), 0)], fr_per_vid=3, frame_diff=True)[0]
    assert clip.shape[0] == 3


def test_video_dataset_raises_when_clip_has_no_frames(tmp_path):
    """An empty clip folder is reported clearly."""
    empty = tmp_path / 'Biking' / 'v_Biking_g01_c01'
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        VideoDataset([(str(empty), 0)], fr_per_vid=4)[0]  # pylint: disable=expression-not-assigned


def test_collate_pads_to_the_longest_clip():
    """Shorter clips are padded with zeros to the longest clip in the batch."""
    batch = [(torch.ones(4, 3, 8, 8), 0), (torch.ones(2, 3, 8, 8), 1)]
    frames, labels = collate_fn_rnn(batch)
    assert frames.shape == (2, 4, 3, 8, 8)
    assert labels.tolist() == [0, 1]
    assert frames[1, 2:].sum() == 0
