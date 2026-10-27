"""
Module: run.py

This module is the main entry point for training or evaluating a video classification model.
It uses command-line arguments to configure the experiment, including dataset paths, model parameters,
and training hyperparameters. Depending on the selected mode ('train' or 'eval'), it performs
the following:

- Train mode:
    - Loads the dataset from a directory structure.
    - Splits the dataset BY GROUP into training, validation, and test sets.
    - Creates PyTorch Datasets and DataLoaders for training and validation.
    - Initializes the model (e.g., LRCN) and defines the loss function, optimizer, and learning rate scheduler.
    - Runs the training loop and saves the best model weights.
    - Logs loss and accuracy for train, validation, and test to Weights & Biases (the test
      numbers are for plotting only and are never used to choose weights).

- Eval mode:
    - Rebuilds the same group-based dataset splits (the split depends only on the group numbers).
    - Creates a DataLoader for the test set.
    - Loads the trained model checkpoint.
    - Evaluates the model on the test set and prints overall test accuracy.

The module also includes a helper function for parsing command-line arguments.
"""

import os
import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import wandb

from video_datasets import (VideoDataset, load_dataset, dataset_split, get_group,
                            DEFAULT_VAL_GROUPS, DEFAULT_TEST_GROUPS)
from utils import (transform_stats, compose_data_transforms, train_val_dloaders, test_dloaders,
                   get_device)
from models import LRCN
from train import train
from evaluate import test, predict_probs, accuracy_by_key, top_confusions

def _log_to_wandb(metrics, step):
    """Send one epoch's metrics to Weights & Biases."""
    wandb.log(metrics, step=step)

def _print_breakdown(split, label_dict, targets, outputs):
    """Print accuracy for each UCF50 group and for the ten weakest classes of a split."""
    names = {label: name for name, label in label_dict.items()}
    by_group = accuracy_by_key([get_group(path) for path, _ in split], targets, outputs)
    by_class = accuracy_by_key([names[target] for target in targets], targets, outputs)
    print('Accuracy by group:')
    for group, acc in sorted(by_group.items()):
        print(f'  group {group:2d}: {100 * acc:.2f}%')
    print('Ten weakest classes:')
    for name, acc in sorted(by_class.items(), key=lambda item: item[1])[:10]:
        print(f'  {name}: {100 * acc:.2f}%')
    print('Most common mistakes (true class -> predicted class):')
    for true, pred, count in top_confusions(targets, outputs, names):
        print(f'  {true} -> {pred}: {count}')

def args_parser():
    """
    Parse command-line arguments for configuring the video classification training or evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments include:
        -fd/--frame_dir: Directory for storing video frames.
        -vg/--val_groups: UCF50 group numbers (1-25) used for validation (default 18 19 20 21).
        -tg/--test_groups: UCF50 group numbers (1-25) used for testing (default 22 23 24 25).
        -fpv/--fr_per_vid: Number of frames per video to consider (default 16).
        -nc/--n_classes: Number of classes for the classification task (required).
        -c/--ckpt: Path for loading a trained model checkpoint.
        -mt/--model_type: Model type (only 'lrcn' is implemented, default 'lrcn').
        -cnn/--cnn_backbone: Backbone CNN for 2D feature extraction (default 'resnet34').
        -p/--pretrained: Whether to use a pretrained CNN backbone (default True).
        -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default 1).
        -m/--mode: Mode of operation: 'train' or 'eval' (required).
        -bs/--batch_size: Mini-batch size (required).
        -d/--dropout: Dropout rate for regularization (default 0.1).
        -lr/--learning_rate: Learning rate for training (default 3e-5).
        -ne/--n_epochs: Number of training epochs (default 30).
        -nw/--num_workers: Background processes used to load data (default 4).
        --wandb/--no-wandb: Log to Weights & Biases (default on).
        --wandb_project: Weights & Biases project name (default 'ucf50-action-recognition').
        --run_name: Optional name for the Weights & Biases run.
        --strong_aug/--no-strong_aug: Stronger training augmentation (default off).
        --label_smoothing: Label smoothing for the training loss (default 0.0).
        --weight_decay: Weight decay of the AdamW optimizer (default 0.0).
        --head_lr: Separate learning rate for the LSTM and final layer (default: same as -lr).
        --scheduler: 'plateau' or 'cosine' learning rate schedule (default 'plateau').
        --temporal_pool: 'last' or 'mean' over the LSTM outputs (default 'last').
        --temporal_jitter/--no-temporal_jitter: Random frame choice per epoch when more frames
            are stored than used (default off).
        --tta/--no-tta: Average each test clip with its mirrored copy (default off).
        --init_ckpt: Start training from the weights in this checkpoint (default: none).
        --eval_split: Which split 'eval' mode scores: 'test' or 'val' (default 'test').
        --frame_diff/--no-frame_diff: Motion model. Read stored frame pairs and use their
            difference as the input (default off).
        --save_probs: In eval mode, save the class probabilities to this .npz file (default: none).
        --report/--no-report: In eval mode, also print accuracy per group and the weakest classes.
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', help='Directory for storing video frames', required=True)
    parser.add_argument('-vg', '--val_groups', type=int, nargs='+', default=list(DEFAULT_VAL_GROUPS),
                        help='UCF50 group numbers (1-25) used for validation')
    parser.add_argument('-tg', '--test_groups', type=int, nargs='+', default=list(DEFAULT_TEST_GROUPS),
                        help='UCF50 group numbers (1-25) used for testing')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16, help='Number of frames per video')
    parser.add_argument('-nc', '--n_classes', type=int, required=True,
                        help='Number of classes for the classification task')

    parser.add_argument('-c', '--ckpt', help='Path for loading trained model checkpoints')
    parser.add_argument('-mt', '--model_type', help='Model type (only lrcn is implemented)', default='lrcn',
                        choices=['lrcn'])
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34',
                        help='2D CNN backbone - options: resnet18, resnet34, resnet50, resnet101, resnet152')
    # BUG FIX: "--pretrained False" used to be read as the (truthy) string "False".
    parser.add_argument('-p', '--pretrained', help='Use pretrained 2D CNN backbone (--no-pretrained to turn off)',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100,
                        help='Number of neurons in the RNN/LSTM hidden layer')
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1, help='Number of RNN/LSTM layers')

    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'eval'],
                        help="Either 'train' or 'eval'", required=True)
    parser.add_argument('-bs', '--batch_size', type=int, help='Mini-batch size', required=True)
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5, help='Learning rate for model training')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('-nw', '--num_workers', type=int, default=4,
                        help='Background processes used to load data')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True,
                        help='Log to Weights & Biases (--no-wandb to turn off)')
    parser.add_argument('--wandb_project', default='ucf50-action-recognition',
                        help='Weights & Biases project name')
    parser.add_argument('--run_name', default=None, help='Name for the Weights & Biases run')

    # Model improvements (all off by default, so the baseline can still be reproduced)
    parser.add_argument('--strong_aug', action=argparse.BooleanOptionalAction, default=False,
                        help='Stronger training augmentation: random resized crop and color jitter')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing for the training loss')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='AdamW weight decay')
    parser.add_argument('--head_lr', type=float, default=None,
                        help='Learning rate for the LSTM and final layer (default: same as --learning_rate)')
    parser.add_argument('--scheduler', choices=['plateau', 'cosine'], default='plateau',
                        help='Learning rate schedule')
    parser.add_argument('--temporal_pool', choices=['last', 'mean'], default='last',
                        help='Use the last LSTM output or the mean over all time steps')

    parser.add_argument('--temporal_jitter', action=argparse.BooleanOptionalAction, default=False,
                        help='Pick different frames each epoch when more frames are stored than used')
    parser.add_argument('--tta', action=argparse.BooleanOptionalAction, default=False,
                        help='Test time augmentation: average each test clip with its mirrored copy')

    parser.add_argument('--frame_diff', action=argparse.BooleanOptionalAction, default=False,
                        help='Motion model: use the difference between stored frame pairs')
    parser.add_argument('--save_probs', default=None,
                        help="In eval mode, save the class probabilities to this .npz file")
    parser.add_argument('--init_ckpt', default=None,
                        help='Start training from the weights in this checkpoint')
    parser.add_argument('--eval_split', choices=['test', 'val'], default='test',
                        help="Which split 'eval' mode scores")
    parser.add_argument('--report', action=argparse.BooleanOptionalAction, default=False,
                        help='In eval mode, also print accuracy per group and the weakest classes')

    return parser.parse_args()

def main(args):  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    """
    Main function to execute training or evaluation based on the parsed command-line arguments.

    For training:
        - Loads the dataset from the specified frame directory.
        - Splits the dataset BY GROUP into training, validation, and test sets.
        - Creates the training and validation DataLoaders.
        - Initializes the model (LRCN) with specified hyperparameters.
        - Defines the loss function, optimizer, and learning rate scheduler.
        - Runs the training loop and saves the best model weights.
        - Logs train/val/test loss and accuracy to Weights & Biases.

    For evaluation:
        - Rebuilds the same group-based dataset splits.
        - Creates the test DataLoader.
        - Loads the trained model checkpoint.
        - Evaluates the model on the test set and prints the overall test accuracy.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Dataset parameters
    frame_dir = args.frame_dir
    val_groups = args.val_groups
    test_groups = args.test_groups
    fr_per_vid = args.fr_per_vid
    n_classes = args.n_classes

    # Model parameters
    model_type = args.model_type
    rnn_hidden_size = args.rnn_hidden_size
    rnn_n_layers = args.rnn_n_layers
    dropout = args.dropout
    pretrained = args.pretrained
    cnn_backbone = args.cnn_backbone
    ckpt = args.ckpt

    # Training parameters
    mode = args.mode
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    num_workers = args.num_workers
    device = get_device()
    print('Using device:', device)

    # Load transformation statistics and create data augmentation transforms
    h, w, mean, std = transform_stats(model_type)
    tr_transforms, val_ts_transforms = compose_data_transforms(h, w, mean, std, args.strong_aug)

    # Load dataset and split BY GROUP. The split only depends on the group numbers, so train
    # and eval mode always get the same test set (no splits file is needed).
    vid_dataset, label_dict = load_dataset(frame_dir)
    if len(label_dict) != n_classes:
        raise ValueError(f'Found {len(label_dict)} classes in {frame_dir} but --n_classes is {n_classes}.')
    tr_split, val_split, ts_split = dataset_split(vid_dataset, val_groups, test_groups)

    # Initialize the model (LRCN)
    model = LRCN(hidden_size=rnn_hidden_size, n_layers=rnn_n_layers, dropout_rate=dropout,
                 n_classes=n_classes, pretrained=pretrained, cnn_model=cnn_backbone,
                 temporal_pool=args.temporal_pool)

    if mode == 'train':
        if args.init_ckpt:
            model.load_state_dict(torch.load(args.init_ckpt, map_location='cpu'))

        # Create PyTorch Datasets and DataLoaders for train and validation
        tr_dataset = VideoDataset(tr_split, fr_per_vid, tr_transforms,
                                  random_sample=args.temporal_jitter, frame_diff=args.frame_diff)
        val_dataset = VideoDataset(val_split, fr_per_vid, val_ts_transforms,
                                   frame_diff=args.frame_diff)
        dataloaders = train_val_dloaders(tr_dataset, val_dataset, batch_size, num_workers)

        # The test set is evaluated every epoch ONLY to draw the test curves. It is never used to
        # choose the best weights (that uses validation accuracy) or to change the learning rate.
        ts_dataset = VideoDataset(ts_split, fr_per_vid, val_ts_transforms,
                                  frame_diff=args.frame_diff)
        dataloaders['test'] = test_dloaders(ts_dataset, batch_size, num_workers)['test']

        # Define the loss function, optimizer, and learning rate scheduler
        loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        eval_loss_func = nn.CrossEntropyLoss()  # plain loss so val/test losses are comparable across runs
        if args.head_lr is None:
            params = model.parameters()
        else:
            # Pretrained CNN gets the (smaller) base rate; the new LSTM and final layer get head_lr
            head_params = list(model.rnn.parameters()) + list(model.fc.parameters())
            params = [{'params': model.base_model.parameters(), 'lr': learning_rate},
                      {'params': head_params, 'lr': args.head_lr}]
        opt = optim.AdamW(params, lr=learning_rate, weight_decay=args.weight_decay)
        if args.scheduler == 'cosine':
            lr_scheduler = CosineAnnealingLR(opt, T_max=n_epochs)
        else:
            lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        os.makedirs("./models", exist_ok=True)
        optim_model_dir = './models'

        # Set up Weights & Biases logging (run "wandb login" once before the first run)
        log_fn = None
        if args.wandb:
            wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
            log_fn = _log_to_wandb

        # Main training procedure
        model.to(device)
        model, loss_hist, acc_hist = train(dataloaders, model, loss_func, opt, lr_scheduler, device,
                                           optim_model_dir, n_epochs, log_fn,
                                           eval_criterion=eval_loss_func)

        # Final test accuracy of the best (by validation accuracy) checkpoint
        _, _, best_test_acc = test(model, dataloaders['test'], device)
        print(f'Test accuracy of the best validation checkpoint: {100 * best_test_acc:.4f}%.')
        if args.tta:
            _, _, tta_test_acc = test(model, dataloaders['test'], device, tta=True)
            print(f'Same checkpoint with flip averaging (TTA): {100 * tta_test_acc:.4f}%.')

        if args.wandb:
            epochs = list(range(1, len(loss_hist['train']) + 1))
            wandb.run.summary['best_val_acc'] = max(acc_hist['val'])
            wandb.run.summary['test_acc_at_best_val_checkpoint'] = best_test_acc
            if args.tta:
                wandb.run.summary['test_acc_tta'] = tta_test_acc
            wandb.log({'loss_curves': wandb.plot.line_series(
                           xs=epochs, ys=[loss_hist['train'], loss_hist['val'], loss_hist['test']],
                           keys=['train', 'val', 'test'], title='Loss', xname='epoch'),
                       'accuracy_curves': wandb.plot.line_series(
                           xs=epochs, ys=[acc_hist['train'], acc_hist['val'], acc_hist['test']],
                           keys=['train', 'val', 'test'], title='Accuracy', xname='epoch')})
            wandb.finish()

    elif mode == 'eval':
        # Create PyTorch Dataset and DataLoader for the chosen split
        eval_split = val_split if args.eval_split == 'val' else ts_split
        ts_dataset = VideoDataset(eval_split, fr_per_vid, val_ts_transforms,
                                  frame_diff=args.frame_diff)
        dataloaders = test_dloaders(ts_dataset, batch_size, num_workers)

        # Load the trained model checkpoint
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        if args.save_probs:
            targets, probs = predict_probs(model, dataloaders['test'], device, tta=args.tta)
            outputs = probs.argmax(axis=1).tolist()
            accuracy = float(np.mean(np.array(targets) == np.array(outputs)))
            np.savez(args.save_probs, targets=np.array(targets), probs=probs)
        else:
            targets, outputs, accuracy = test(model, dataloaders['test'], device, tta=args.tta)

        print(f'The overall {args.eval_split} accuracy is {100 * accuracy:.4f}%.')
        if args.report:
            _print_breakdown(eval_split, label_dict, targets, outputs)

    else:
        raise ValueError('The mode argument must be either "train" or "eval".')

if __name__ == "__main__":
    main(args_parser())
