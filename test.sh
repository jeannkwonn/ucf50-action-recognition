#!/bin/bash
# Evaluate the baseline checkpoint on the held out test groups. 
# For the final blended result use eval_fuse.sh.
python run.py \
    --frame_dir ../ucf50-data/ucf50_frames \
    --val_groups 18 19 20 21 \
    --test_groups 22 23 24 25 \
    --ckpt ./models_baseline/best_model_wts.pt \
    --model_type lrcn \
    --n_classes 50 \
    --batch_size 4 \
    --eval_split test \
    --mode eval