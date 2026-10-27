#!/bin/bash
# Improved run v3: ResNet50 backbone, different frames each epoch (temporal jitter),
# AdamW with separate learning rates, cosine schedule, mean pooling, flip averaging at test time.
# Needs 32 frames per video: run preprocess.py with --n_frames 32 into ucf50_frames32 first.
python run.py \
    --frame_dir ../ucf50-data/ucf50_frames32 \
    --val_groups 18 19 20 21 \
    --test_groups 22 23 24 25 \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --n_epochs 10 \
    --cnn_backbone resnet50 \
    --rnn_hidden_size 256 \
    --learning_rate 3e-5 \
    --head_lr 3e-4 \
    --weight_decay 0.01 \
    --scheduler cosine \
    --label_smoothing 0.1 \
    --dropout 0.3 \
    --temporal_pool mean \
    --temporal_jitter \
    --tta \
    --run_name improved-v3 \
    --mode train