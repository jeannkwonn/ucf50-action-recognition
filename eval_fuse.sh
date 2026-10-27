#!/bin/bash
# Score the picture model (v3) and the motion model on the validation and test groups,
# save their probabilities, and blend them. The blend weight is chosen on validation only.
COMMON="--val_groups 18 19 20 21 --test_groups 22 23 24 25 --model_type lrcn --n_classes 50 \
    --fr_per_vid 16 --batch_size 4 --mode eval --tta --rnn_hidden_size 256 --temporal_pool mean"
for split in val test; do
    python run.py --frame_dir ../ucf50-data/ucf50_frames32 $COMMON --cnn_backbone resnet50 \
        --ckpt models_v3_partial/best_model_wts.pt --eval_split $split \
        --save_probs probs_picture_$split.npz
    python run.py --frame_dir ../ucf50-data/ucf50_pairs $COMMON --cnn_backbone resnet34 \
        --frame_diff --ckpt models_motion/best_model_wts.pt --eval_split $split \
        --save_probs probs_motion_$split.npz
done
python fuse.py --val probs_picture_val.npz probs_motion_val.npz \
    --test probs_picture_test.npz probs_motion_test.npz --wandb
        