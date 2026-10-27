"""
Module: fuse.py

Blend the predictions of two saved models (for example the picture model and the motion model).

Each model first saves its class probabilities with run.py in eval mode, using --save_probs.
This script then averages the two sets of probabilities with a weight and reports accuracy.
The weight is chosen on the VALIDATION split only, and then applied unchanged to the test
split, so the test set never influences any choice.

Usage:
    python fuse.py --val rgb_val.npz motion_val.npz --test rgb_test.npz motion_test.npz

Add --wandb to also save the blended result as a small Weights & Biases run.
"""

import argparse

import numpy as np
import wandb


def load_probs(path):
    """
    Load saved predictions.

    Args:
        path (str): Path of an .npz file written by run.py with --save_probs.

    Returns:
        tuple: (targets, probs) as numpy arrays.
    """
    data = np.load(path)
    return data['targets'], data['probs']


def accuracy(targets, probs):
    """Return the fraction of samples whose most likely class equals the target."""
    return float(np.mean(probs.argmax(axis=1) == targets))


def blend(probs_a, probs_b, weight):
    """Return weight * probs_a + (1 - weight) * probs_b."""
    return weight * probs_a + (1 - weight) * probs_b


def best_weight(targets, probs_a, probs_b, step=0.05):
    """
    Find the blend weight with the highest accuracy.

    Args:
        targets (numpy.ndarray): Ground truth labels.
        probs_a (numpy.ndarray): Probabilities of model A.
        probs_b (numpy.ndarray): Probabilities of model B.
        step (float, optional): Spacing of the weights tried between 0 and 1. Default is 0.05.

    Returns:
        float: The weight for model A. Ties go to the weight closest to 0.5.
    """
    weights = np.arange(0.0, 1.0 + step / 2, step)
    scored = [(accuracy(targets, blend(probs_a, probs_b, w)), -abs(w - 0.5), w) for w in weights]
    return float(max(scored)[2])


def args_parser():
    """Parse the command-line arguments (two validation files and two test files)."""
    parser = argparse.ArgumentParser(description='Blend the predictions of two models')
    parser.add_argument('--val', nargs=2, required=True, metavar=('A_VAL', 'B_VAL'),
                        help='Validation predictions of model A and model B')
    parser.add_argument('--test', nargs=2, required=True, metavar=('A_TEST', 'B_TEST'),
                        help='Test predictions of model A and model B')
    parser.add_argument('--wandb', action='store_true',
                        help='Log the blend weight and accuracies to Weights & Biases')
    parser.add_argument('--wandb_project', default='ucf50-action-recognition',
                        help='Weights & Biases project name')
    parser.add_argument('--run_name', default='blend-final', help='Weights & Biases run name')
    return parser.parse_args()


def main(args):  # pylint: disable=too-many-locals
    """Choose the blend weight on validation, then report validation and test accuracy."""
    val_targets, val_a = load_probs(args.val[0])
    val_targets_b, val_b = load_probs(args.val[1])
    test_targets, test_a = load_probs(args.test[0])
    test_targets_b, test_b = load_probs(args.test[1])
    if not (np.array_equal(val_targets, val_targets_b) and np.array_equal(test_targets, test_targets_b)):
        raise ValueError('The two models were scored on different videos or in a different order.')

    weight = best_weight(val_targets, val_a, val_b)
    print(f'Weight for model A chosen on validation: {weight:.2f} (model B gets {1 - weight:.2f})')
    results = {'weight_picture_model': weight}
    for name, targets, probs_a, probs_b in (('val', val_targets, val_a, val_b),
                                            ('test', test_targets, test_a, test_b)):
        acc_a = 100 * accuracy(targets, probs_a)
        acc_b = 100 * accuracy(targets, probs_b)
        acc_blend = 100 * accuracy(targets, blend(probs_a, probs_b, weight))
        print(f'{name}: A alone {acc_a:.2f}%, B alone {acc_b:.2f}%, blended {acc_blend:.2f}%')
        results.update({f'{name}_acc_model_a': acc_a, f'{name}_acc_model_b': acc_b,
                        f'{name}_acc_blend': acc_blend})

    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb.run.summary.update(results)
        wandb.finish()


if __name__ == '__main__':
    main(args_parser())
