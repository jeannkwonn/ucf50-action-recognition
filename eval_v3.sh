#!/bin/bash
# Score the saved v3 checkpoint on the VALIDATION groups, with accuracy per group and per class.
# Use --eval_split test for the final test score.
python run.py \
    --frame_dir ../ucf50-data/ucf50_frames32 \
    --val_groups 18 19 20 21 \
    --test_groups 22 23 24 25 \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --cnn_backbone resnet50 \
    --rnn_hidden_size 256 \
    --temporal_pool mean \
    --ckpt models_v3_partial/best_model_wts.pt \
    --eval_split val \
    --report \
    --mode eval