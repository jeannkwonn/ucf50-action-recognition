#!/bin/bash
# Motion model: the same LRCN, but it sees the DIFFERENCE between two nearby frames,
# so it learns what moves instead of what the scene looks like.
# Needs frame pairs first:
#   python preprocess.py --video_dir ../ucf50-data/UCF50 --frame_dir ../ucf50-data/ucf50_pairs \
#       --n_frames 16 --pair_gap 4
python run.py \
    --frame_dir ../ucf50-data/ucf50_pairs \
    --val_groups 18 19 20 21 \
    --test_groups 22 23 24 25 \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --n_epochs 10 \
    --cnn_backbone resnet34 \
    --rnn_hidden_size 256 \
    --learning_rate 5e-5 \
    --head_lr 5e-4 \
    --weight_decay 0.01 \
    --scheduler cosine \
    --label_smoothing 0.1 \
    --dropout 0.3 \
    --temporal_pool mean \
    --frame_diff \
    --tta \
    --run_name motion-v1 \
    --mode train