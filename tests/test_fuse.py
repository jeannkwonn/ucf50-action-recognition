"""Unit tests for fuse.py: blending the predictions of two models."""

import numpy as np
import pytest

from fuse import accuracy, best_weight, blend, load_probs


def test_accuracy_counts_correct_argmax():
    """Two of the three samples are predicted correctly."""
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    assert accuracy(np.array([0, 1, 1]), probs) == pytest.approx(2 / 3)


def test_blend_is_a_weighted_average():
    """A weight of 0.25 gives a quarter of model A and three quarters of model B."""
    result = blend(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), 0.25)
    assert result.tolist() == [[0.25, 0.75]]


def test_best_weight_prefers_the_model_that_is_right():
    """If only model A is right, the best weight leans entirely on A."""
    targets = np.array([0, 1, 0])
    probs_a = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    probs_b = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    assert best_weight(targets, probs_a, probs_b) > 0.5


def test_best_weight_ties_go_to_the_middle():
    """If both models give the same accuracy at every weight, 0.5 is chosen."""
    targets = np.array([0, 1])
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    assert best_weight(targets, probs, probs) == pytest.approx(0.5)


def test_load_probs_round_trip(tmp_path):
    """Saved predictions are read back unchanged."""
    path = tmp_path / 'p.npz'
    np.savez(path, targets=np.array([1, 0]), probs=np.array([[0.2, 0.8], [0.7, 0.3]]))
    targets, probs = load_probs(str(path))
    assert targets.tolist() == [1, 0]
    assert probs.shape == (2, 2)
