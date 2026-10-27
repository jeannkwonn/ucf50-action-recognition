"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as directories
of frame images, functions to load the dataset from a directory structure, split the dataset
by group (so near-duplicate clips never end up in different splits), and a custom collate
function for handling variable-length video sequences.
"""

import os
import re
import glob

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import functional as TF

# UCF50 clip names look like v_Biking_g07_c03: "g07" is the group, "c03" the clip in the group.
# Default group split: groups 1-17 train, 18-21 validation, 22-25 test.
DEFAULT_VAL_GROUPS = (18, 19, 20, 21)
DEFAULT_TEST_GROUPS = (22, 23, 24, 25)

_GROUP_PATTERN = re.compile(r'_g(\d+)_')
_DIGITS_PATTERN = re.compile(r'(\d+)')


def natural_sort_key(path):
    """
    Build a sort key that orders file names by their numbers, not alphabetically.

    Alphabetical order puts frame10.jpg before frame2.jpg. This key sorts the numbers
    as numbers, so frames come back in the order they were recorded.

    Args:
        path (str): File path or name.

    Returns:
        list: Mixed list of strings and integers that can be used as a sort key.
    """
    return [int(part) if part.isdigit() else part for part in _DIGITS_PATTERN.split(path)]


def get_group(vid_path):
    """
    Read the UCF50 group number from a clip name.

    Args:
        vid_path (str): Path to a clip, for example ".../v_Biking_g07_c03".

    Returns:
        int: The group number (7 in the example).

    Raises:
        ValueError: If the name has no "_gXX_" group tag.
    """
    match = _GROUP_PATTERN.search(os.path.basename(os.path.normpath(vid_path)))
    if match is None:
        raise ValueError(f'Cannot find a group tag like "_g01_" in "{vid_path}".')
    return int(match.group(1))


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.

    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a list of (video directory path, label) tuples.

    Args:
        vid_dataset (list): List of (video directory path, integer label) tuples.
        fr_per_vid (int): Number of frames per video to load (images are taken in order).
        random_sample (bool, optional): If True and a video has more stored frames than fr_per_vid,
            pick one random frame from each of fr_per_vid equal time segments, so every epoch sees
            slightly different frames (temporal jitter). If False, always pick the same evenly
            spaced frames. Default is False. Use True for training only.
        frame_diff (bool, optional): If True, the stored frames are read as pairs (first, second,
            first, second, ...) and each clip frame is the difference second - first, so the
            model sees motion instead of appearance. Default is False.
        transforms (callable, optional): A transform applied to the whole clip tensor of shape
            (T, C, H, W), e.g., resizing, normalization. Because it sees the whole clip, random
            augmentations (such as a flip) are applied the same way to every frame of that clip.
    """
    def __init__(self, vid_dataset, fr_per_vid, transforms=None, random_sample=False,
                 frame_diff=False):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms
        self.random_sample = random_sample
        self.frame_diff = frame_diff

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def _choose_frames(self, n_available):
        """
        Choose which frames of a video to load.

        Args:
            n_available (int): Number of stored frames (more than fr_per_vid).

        Returns:
            list: fr_per_vid frame indices in time order. Evenly spaced when random_sample is
                False. One random index per equal time segment when random_sample is True.
        """
        if not self.random_sample:
            return np.linspace(0, n_available - 1, self.fpv).round().astype(int).tolist()
        edges = np.linspace(0, n_available, self.fpv + 1)
        low = np.floor(edges[:-1]).astype(int)
        high = np.maximum(np.floor(edges[1:]).astype(int), low + 1)
        return [int(torch.randint(int(lo), int(hi), (1,))) for lo, hi in zip(low, high)]

    def _pair_paths(self, fr_paths):
        """Group stored frames into (first, second) pairs and keep fr_per_vid of the pairs."""
        n_pairs = len(fr_paths) // 2
        keep = self._choose_frames(n_pairs) if n_pairs > self.fpv else range(n_pairs)
        return [fr_paths[2 * i + j] for i in keep for j in (0, 1)]

    def __getitem__(self, idx):
        """
        Load frames from the video directory corresponding to the given index, apply transforms,
        and return the stacked tensor of frames along with its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (frames_tensor, label) where frames_tensor is a tensor of shape (T, C, H, W)
                   with T being the number of frames (up to fr_per_vid) and label is an integer.
        """
        vid_path, fr_label = self.dataset[idx]

        # Get all JPEG frame paths from the video directory, in time order.
        # BUG FIX: glob returns files in arbitrary order, so sort them by frame number.
        fr_paths = sorted(glob.glob(vid_path + '/*.jpg'), key=natural_sort_key)
        if not fr_paths:
            raise FileNotFoundError(f'No .jpg frames found in "{vid_path}".')

        # Select fr_per_vid frames (or frame pairs) if the video has more than that
        if self.frame_diff:
            fr_paths = self._pair_paths(fr_paths)
        elif len(fr_paths) > self.fpv:
            fr_paths = [fr_paths[i] for i in self._choose_frames(len(fr_paths))]

        # Open images using PIL and convert each one to a tensor
        fr_imgs = []
        for fr_path in fr_paths:
            with Image.open(fr_path) as img:
                fr_imgs.append(TF.to_tensor(img.convert('RGB')))
        clip = torch.stack(fr_imgs)

        # Apply transforms to the whole clip so random augmentations match across frames
        if self.transforms:
            clip = self.transforms(clip)

        # Motion model: keep only what changed between the two frames of each pair
        if self.frame_diff:
            clip = clip[1::2] - clip[0::2]

        return clip, int(fr_label)


def load_dataset(frame_dir):
    """
    Load the full video dataset from the specified directory.

    Each subdirectory in frame_dir is assumed to correspond to a video category.
    The function builds a dictionary where keys are paths to video directories and
    values are integer labels corresponding to each category.

    Args:
        frame_dir (str): Path to the directory containing subdirectories for each video category.

    Returns:
        tuple: (vid_dataset, label_dict)
            - vid_dataset (dict): Dictionary mapping video directory paths to integer labels.
            - label_dict (dict): Dictionary mapping video category names to integer labels.
    """
    # Only real folders count (skips files such as macOS ".DS_Store")
    classes = sorted(c for c in os.listdir(frame_dir) if os.path.isdir(os.path.join(frame_dir, c)))
    label_dict = {vid_cat: idx for idx, vid_cat in enumerate(classes)}
    vid_dataset = {}
    print('Loading video dataset....')
    for vid_cat in tqdm(classes):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        for vid in sorted(os.listdir(vid_cat_path)):
            vid_path = os.path.join(vid_cat_path, vid)
            if os.path.isdir(vid_path):
                vid_dataset[vid_path] = label_dict[vid_cat]
    return vid_dataset, label_dict


def dataset_split(vid_dataset, val_groups=DEFAULT_VAL_GROUPS, test_groups=DEFAULT_TEST_GROUPS):
    """
    Split the dataset into training, validation, and test sets BY GROUP.

    DATA LEAK FIX: the old code shuffled individual clips, so clips from the same group (same
    person, background, and camera) landed in both train and test. Here every group belongs to
    exactly one split, so the test set only contains groups the model never saw in training.

    Args:
        vid_dataset (dict): Dictionary mapping video paths to labels.
        val_groups (iterable): Group numbers (1-25) used for validation.
        test_groups (iterable): Group numbers (1-25) used for testing.

    Returns:
        tuple: (tr_dataset, val_dataset, ts_dataset)
            - tr_dataset (list): List of (video_path, label) tuples for the training set.
            - val_dataset (list): List of (video_path, label) tuples for the validation set.
            - ts_dataset (list): List of (video_path, label) tuples for the test set.
            Groups that are not in val_groups or test_groups go to the training set.

    Raises:
        ValueError: If the validation and test groups overlap, or any split would be empty.
    """
    val_groups, test_groups = set(val_groups), set(test_groups)
    if val_groups & test_groups:
        raise ValueError(f'Validation and test groups overlap: {sorted(val_groups & test_groups)}')

    print('Splitting train/validation/test datasets by group....')
    splits = {'train': [], 'val': [], 'test': []}
    for vid_path, vid_label in vid_dataset.items():
        group = get_group(vid_path)
        if group in test_groups:
            splits['test'].append((vid_path, vid_label))
        elif group in val_groups:
            splits['val'].append((vid_path, vid_label))
        else:
            splits['train'].append((vid_path, vid_label))

    for name, items in splits.items():
        if not items:
            raise ValueError(f'The {name} split is empty. Check the group numbers.')
    return splits['train'], splits['val'], splits['test']


def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models.

    Handles variable-length video sequences by padding them to the length of the longest sequence
    in the batch. Each sample in the batch is expected to be a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). The function returns a padded tensor
    of video frames with shape (batch_size, max_T, C, H, W) and a tensor of labels.

    Args:
        batch (list): List of samples, each as (video_frames, label).

    Returns:
        tuple: (padded_imgs, labels_tensor)
            - padded_imgs (Tensor): Padded tensor of video frames.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Unzip the batch into image tensors and labels
    imgs_batch, label_batch = list(zip(*batch))

    # Filter out any samples that have no frames
    valid_samples = [(imgs, label) for imgs, label in zip(imgs_batch, label_batch) if len(imgs) > 0]
    if not valid_samples:
        return None, None
    imgs_batch, label_batch = zip(*valid_samples)

    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)

    return padded_imgs, labels_tensor
