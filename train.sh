#!/bin/bash
# Train on UCF50. Run preprocess.py first (see README) to create the frame folders.
python run.py \
    --frame_dir ../ucf50-data/ucf50_frames \
    --val_groups 18 19 20 21 \
    --test_groups 22 23 24 25 \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --n_epochs 12 \
    --run_name baseline-fixed \
    --mode train