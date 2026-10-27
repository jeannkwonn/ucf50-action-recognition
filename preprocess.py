"""
Module: preprocess.py

Convert the raw UCF50 videos (.avi) into folders of JPEG frames that the training code reads.

Input layout:   <video_dir>/<ClassName>/v_<ClassName>_gXX_cYY.avi
Output layout:  <frame_dir>/<ClassName>/v_<ClassName>_gXX_cYY/frame_000.jpg ... frame_015.jpg

Usage:
    python preprocess.py --video_dir ../ucf50-data/UCF50 --frame_dir ../ucf50-data/ucf50_frames

With --pair_gap G (G > 0) the script instead saves PAIRS of nearby frames for a motion model:
for each of the n_frames positions it stores a frame and the frame G frames later, so each
video gets 2 * n_frames files (frame_000.jpg, frame_001.jpg = first pair, and so on).

Videos that were already converted are skipped, so the script can be stopped and restarted.
"""

import argparse
import glob
import os

from tqdm import tqdm

from utils import get_frames, get_frame_pairs, store_frames


def args_parser():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments (video_dir, frame_dir, n_frames).
    """
    parser = argparse.ArgumentParser(description='Extract frames from UCF50 videos')
    parser.add_argument('-vd', '--video_dir', required=True,
                        help='Folder containing one subfolder of .avi files per action class')
    parser.add_argument('-fd', '--frame_dir', required=True,
                        help='Output folder for the extracted frames')
    parser.add_argument('-nf', '--n_frames', type=int, default=16,
                        help='Number of evenly spaced frames (or frame pairs) to keep per video')
    parser.add_argument('-pg', '--pair_gap', type=int, default=0,
                        help='If above 0, store pairs of frames this many frames apart (for motion)')
    return parser.parse_args()


def preprocess(video_dir, frame_dir, n_frames=16, pair_gap=0):
    """
    Extract ``n_frames`` evenly spaced frames from every video and save them as JPEGs.

    Args:
        video_dir (str): Folder with one subfolder of videos per class.
        frame_dir (str): Output folder.
        n_frames (int): Number of frames (or frame pairs) to keep per video.
        pair_gap (int): If above 0, store a pair of frames pair_gap frames apart at each of the
            n_frames positions, giving 2 * n_frames files per video.

    Returns:
        tuple: (n_done, n_skipped, failed) - counts of converted and already-done videos, and
            a list of video paths that could not be read.
    """
    classes = sorted(c for c in os.listdir(video_dir)
                     if os.path.isdir(os.path.join(video_dir, c)))
    n_done, n_skipped, failed = 0, 0, []
    for cls in classes:
        videos = sorted(glob.glob(os.path.join(video_dir, cls, '*.avi')))
        for vid in tqdm(videos, desc=cls):
            out_dir = os.path.join(frame_dir, cls, os.path.splitext(os.path.basename(vid))[0])
            n_files = 2 * n_frames if pair_gap > 0 else n_frames
            if len(glob.glob(os.path.join(out_dir, '*.jpg'))) == n_files:
                n_skipped += 1
                continue
            if pair_gap > 0:
                frames, _ = get_frame_pairs(vid, n_frames, pair_gap)
            else:
                frames, _ = get_frames(vid, n_frames)
            if not frames:
                failed.append(vid)
                continue
            store_frames(frames, out_dir)
            n_done += 1
    return n_done, n_skipped, failed


if __name__ == '__main__':
    ARGS = args_parser()
    DONE, SKIPPED, FAILED = preprocess(ARGS.video_dir, ARGS.frame_dir, ARGS.n_frames,
                                     ARGS.pair_gap)
    print(f'Converted {DONE} videos, skipped {SKIPPED} already done, {len(FAILED)} failed.')
    for path in FAILED:
        print('  failed:', path)
