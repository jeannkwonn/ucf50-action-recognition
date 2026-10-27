"""
Module: utils.py

This module provides helper functions for video processing and data transformations
for video classification tasks. It includes functions for:
    - Uniformly sampling frames from videos, or pairs of nearby frames (for motion).
    - Storing extracted frames as JPEG images.
    - Retrieving image transformation statistics based on the model type.
    - Composing data transforms for training and validation/test datasets.
    - Creating DataLoaders for training, validation, and testing, using a custom collate function.
    - Choosing the compute device (GPU, Apple MPS, or CPU).
"""

import os
import cv2
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from video_datasets import collate_fn_rnn


def get_device():
    """
    Pick the best available compute device.

    Returns:
        torch.device: CUDA GPU if available, else Apple MPS if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_frames(vid, n_frames=1):
    """
    Uniformly sample frames from a video file.

    Args:
        vid (str): Path to the video file.
        n_frames (int): Number of frames to sample from the video.

    Returns:
        tuple: (frames, v_len)
            - frames (list): List of sampled frames (as numpy arrays in RGB format).
            - v_len (int): Total number of frames in the video.

    Notes:
        - If the video cannot be opened or contains no frames, an empty list and 0 are returned.
        - Frames are sampled at uniformly spaced indices.
        - Exactly n_frames frames are returned. If a frame cannot be decoded, the nearest
          decoded frame is used instead.
    """
    v_cap = cv2.VideoCapture(vid)
    if not v_cap.isOpened():
        print("Failed to open video:", vid)
        return [], 0
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if v_len <= 0:
        print("No frames found in video:", vid)
        v_cap.release()
        return [], 0

    # BUG FIX: the old code asked linspace for n_frames+1 points, so it returned 17 frames
    # when 16 were requested.
    wanted = np.linspace(0, v_len - 1, n_frames).round().astype(int)
    needed = set(wanted.tolist())
    grabbed = {}
    for idx in range(int(wanted.max()) + 1):
        success, frame = v_cap.read()
        if not success:
            break
        if idx in needed:
            grabbed[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    v_cap.release()

    if not grabbed:
        print("Could not decode video:", vid)
        return [], v_len
    frames = []
    for idx in wanted:
        key = idx if idx in grabbed else min(grabbed, key=lambda k, i=idx: abs(k - i))
        frames.append(grabbed[key])
    return frames, v_len


def get_frame_pairs(vid, n_pairs=1, gap=4):
    """
    Sample pairs of nearby frames from a video, to capture motion.

    The pairs start at evenly spaced positions. Each pair is a frame and the frame ``gap``
    frames later, so the difference between the two shows what moved.

    Args:
        vid (str): Path to the video file.
        n_pairs (int): Number of pairs to sample.
        gap (int): Distance in frames between the two frames of a pair.

    Returns:
        tuple: (frames, v_len)
            - frames (list): 2 * n_pairs frames (RGB numpy arrays) in the order
              first_0, second_0, first_1, second_1, ...
            - v_len (int): Total number of frames in the video.

    Notes:
        - If the video cannot be opened or is too short for a pair, an empty list is returned.
        - If a frame cannot be decoded, the nearest decoded frame is used instead.
    """
    v_cap = cv2.VideoCapture(vid)
    if not v_cap.isOpened():
        print("Failed to open video:", vid)
        return [], 0
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if v_len <= gap:
        print("Video too short for frame pairs:", vid)
        v_cap.release()
        return [], 0

    starts = np.linspace(0, v_len - 1 - gap, n_pairs).round().astype(int)
    needed = set(starts.tolist()) | set((starts + gap).tolist())
    grabbed = {}
    for idx in range(max(needed) + 1):
        success, frame = v_cap.read()
        if not success:
            break
        if idx in needed:
            grabbed[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    v_cap.release()

    if not grabbed:
        print("Could not decode video:", vid)
        return [], v_len
    frames = []
    for start in starts.tolist():
        for idx in (start, start + gap):
            key = idx if idx in grabbed else min(grabbed, key=lambda k, i=idx: abs(k - i))
            frames.append(grabbed[key])
    return frames, v_len


def store_frames(frames, store_path):
    """
    Save a list of frames as JPEG images to the specified directory.

    Each frame is converted from RGB to BGR format (as expected by OpenCV)
    before saving. File names are zero padded (frame_000.jpg, frame_001.jpg, ...) so that
    sorting them alphabetically also sorts them in time order.

    Args:
        frames (list): List of frames (numpy arrays in RGB format) to save.
        store_path (str): Directory path where the frames will be stored.

    Returns:
        None
    """
    os.makedirs(store_path, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path_to_frame = os.path.join(store_path, f"frame_{idx:03d}.jpg")
        cv2.imwrite(path_to_frame, frame)


def transform_stats(model='lrcn'):
    """
    Retrieve transformation statistics based on the model type.

    For the 'lrcn' model, images are resized to 224x224; for '3dcnn', images are resized to 112x112.
    Also returns the mean and standard deviation values used for normalization.

    Args:
        model (str): Type of model ('lrcn' or '3dcnn').

    Returns:
        tuple: (h, w, mean, std)
            - h (int): Image height.
            - w (int): Image width.
            - mean (list): Mean values for normalization.
            - std (list): Standard deviation values for normalization.

    Raises:
        ValueError: If an undefined model type is provided.
    """
    if model == 'lrcn':
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model == '3dcnn':
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    else:
        raise ValueError('model_type arg is undefined....')
    return h, w, mean, std


def compose_data_transforms(height, width, mean, std, strong_aug=False):
    """
    Compose and return data transforms for training and validation/test datasets.

    The training transforms include data augmentation such as random horizontal flipping and
    random affine transformations, while the validation/test transforms consist solely of
    resizing and normalizing.

    The transforms act on a whole clip tensor of shape (T, C, H, W) (the dataset already converts
    the frames to tensors), so a random flip or shift is applied identically to every frame of a
    clip instead of flickering from frame to frame.

    Args:
        height (int): Desired image height.
        width (int): Desired image width.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.
        strong_aug (bool, optional): If True, the training transforms use stronger augmentation
            (random resized crop and color jitter) to reduce overfitting. Default is False.

    Returns:
        tuple: (train_transforms, val_test_transforms)
            - train_transforms: Composed transforms for the training set.
            - val_test_transforms: Composed transforms for the validation/test set.
    """
    if strong_aug:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop((height, width), scale=(0.5, 1.0), ratio=(0.75, 1.33),
                                         antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((height, width), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Normalize(mean, std),
        ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((height, width), antialias=True),
        transforms.Normalize(mean, std),
    ])
    return train_transforms, val_test_transforms


def train_val_dloaders(train_dataset, val_dataset, batch_size, num_workers=0):
    """
    Create DataLoaders for training and validation datasets.

    Uses collate_fn_rnn, which pads sequences to equal lengths.

    Args:
        train_dataset (Dataset): PyTorch Dataset for training data.
        val_dataset (Dataset): PyTorch Dataset for validation data.
        batch_size (int): Number of samples per batch.
        num_workers (int): Background processes used to load data (0 loads in the main process).

    Returns:
        dict: Dictionary with keys 'train' and 'val' mapping to their respective DataLoaders.
    """
    kwargs = {'collate_fn': collate_fn_rnn, 'num_workers': num_workers,
              'persistent_workers': num_workers > 0}
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_dl = DataLoader(val_dataset, batch_size=2 * batch_size, shuffle=False, **kwargs)
    dataloaders = {'train': train_dl, 'val': val_dl}
    return dataloaders


def test_dloaders(test_dataset, batch_size, num_workers=0):
    """
    Create a DataLoader for the test dataset.

    Uses collate_fn_rnn, which pads sequences to equal lengths.

    Args:
        test_dataset (Dataset): PyTorch Dataset for test data.
        batch_size (int): Number of samples per batch.
        num_workers (int): Background processes used to load data (0 loads in the main process).

    Returns:
        dict: Dictionary with key 'test' mapping to the test DataLoader.
    """
    test_dl = DataLoader(test_dataset, batch_size=2 * batch_size, shuffle=False,
                         collate_fn=collate_fn_rnn, num_workers=num_workers,
                         persistent_workers=num_workers > 0)
    dataloaders = {'test': test_dl}
    return dataloaders
