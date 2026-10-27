"""Unit tests for train.py and evaluate.py: loss averaging, training loop, and evaluation."""

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

import evaluate
from train import batch_correct_preds, get_epoch_loss, get_learning_rate, train

CPU = torch.device('cpu')


def make_loader(n_samples=6, batch_size=4, shuffle=False):
    """Tiny classification data: each sample is a one hot vector whose label is its position."""
    labels = torch.arange(n_samples) % 3
    inputs = torch.nn.functional.one_hot(labels, 3).float()
    return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=shuffle)


def test_get_learning_rate_reads_the_optimizer():
    """The first parameter group's learning rate is returned."""
    optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.123)
    assert get_learning_rate(optimizer) == 0.123


def test_batch_correct_preds_counts_matches():
    """Two of the three predictions below match the targets."""
    output = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    assert batch_correct_preds(output, torch.tensor([0, 1, 1])) == 2


def test_epoch_loss_is_a_true_average_over_uneven_batches():
    """With 6 samples and batch size 4 the batches have sizes 4 and 2. The epoch loss must still
    equal the plain average over all 6 samples."""
    torch.manual_seed(0)
    model = nn.Linear(3, 3)
    loader = make_loader(n_samples=6, batch_size=4)
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        loss, accuracy = get_epoch_loss(model, criterion, loader, CPU)
        inputs, labels = loader.dataset.tensors
        expected = criterion(model(inputs), labels).item()
        expected_acc = (model(inputs).argmax(dim=1) == labels).float().mean().item()
    assert abs(loss - expected) < 1e-5
    assert abs(accuracy - expected_acc) < 1e-6


def run_training(scheduler_name, tmp_path, with_test_loader):
    """Train a tiny model for 2 epochs and return the model, history, and logged metrics."""
    torch.manual_seed(0)
    model = nn.Linear(3, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    if scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=2)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
    loaders = {'train': make_loader(shuffle=True), 'val': make_loader()}
    if with_test_loader:
        loaders['test'] = make_loader()
    logged = []
    result = train(loaders, model, nn.CrossEntropyLoss(), optimizer, scheduler, CPU,
                   str(tmp_path), n_epochs=2, log_fn=lambda metrics, step: logged.append(metrics))
    return result, logged


def test_train_saves_best_weights_and_logs_every_epoch(tmp_path):
    """The loop saves a checkpoint, fills the histories, and calls the logger once per epoch."""
    (_, loss_hist, acc_hist), logged = run_training('plateau', tmp_path, with_test_loader=True)
    assert (tmp_path / 'best_model_wts.pt').exists()
    assert len(loss_hist['train']) == len(loss_hist['val']) == len(loss_hist['test']) == 2
    assert len(acc_hist['val']) == 2
    assert len(logged) == 2
    assert {'train/loss', 'val/acc', 'test/loss', 'test/acc', 'learning_rate'} <= set(logged[0])


def test_train_works_without_a_test_loader(tmp_path):
    """The test loader is optional, and no test numbers are logged without it."""
    (_, loss_hist, _), logged = run_training('cosine', tmp_path, with_test_loader=False)
    assert not loss_hist['test']
    assert 'test/acc' not in logged[0]


def test_cosine_schedule_lowers_the_learning_rate(tmp_path):
    """The learning rate logged in epoch 2 is lower than in epoch 1."""
    _, logged = run_training('cosine', tmp_path, with_test_loader=False)
    assert logged[1]['learning_rate'] < logged[0]['learning_rate']


def test_evaluate_reports_accuracy_and_labels():
    """A model that copies its one hot input is always right, so accuracy is 100%."""
    model = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(3))
    targets, outputs, accuracy = evaluate.test(model, make_loader(), CPU)
    assert accuracy == 1.0
    assert targets == outputs == [0, 1, 2, 0, 1, 2]


class LeftRightModel(nn.Module):
    """Test model whose scores change when the clip is mirrored: left half minus right half."""
    def forward(self, x):  # pylint: disable=arguments-differ
        """Return one score per sample for two classes."""
        half = x.shape[-1] // 2
        diff = x[..., :half].mean(dim=(1, 2, 3, 4)) - x[..., half:].mean(dim=(1, 2, 3, 4))
        return torch.stack([diff, -diff], dim=1)


def test_flip_averaging_averages_clip_and_mirror():
    """The clip is bright on the right, so the plain model picks class 1. Averaging with the
    mirrored clip cancels the left/right difference, which leaves a tie that goes to class 0."""
    clip = torch.zeros(1, 2, 1, 2, 4)
    clip[..., 2:] = 1.0
    data = DataLoader(TensorDataset(clip, torch.tensor([0])), batch_size=1)
    _, _, plain_accuracy = evaluate.test(LeftRightModel(), data, CPU)
    _, _, tta_accuracy = evaluate.test(LeftRightModel(), data, CPU, tta=True)
    assert plain_accuracy == 0.0
    assert tta_accuracy == 1.0


def test_predict_probs_returns_probabilities_that_sum_to_one():
    """Each row of probabilities sums to 1, and the labels come back in order."""
    model = nn.Linear(3, 3)
    targets, probs = evaluate.predict_probs(model, make_loader(), CPU)
    assert targets == [0, 1, 2, 0, 1, 2]
    assert probs.shape == (6, 3)
    assert torch.allclose(torch.tensor(probs).sum(dim=1), torch.ones(6), atol=1e-5)


def test_accuracy_by_key_scores_each_key_separately():
    """Accuracy is computed per key, here per class name."""
    keys = ['a', 'a', 'b', 'b', 'b']
    result = evaluate.accuracy_by_key(keys, [0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
    assert result == {'a': 0.5, 'b': 2 / 3}


def test_top_confusions_lists_the_most_common_mistakes():
    """Mistakes are counted per (true, predicted) pair and the most frequent comes first."""
    names = {0: 'a', 1: 'b', 2: 'c'}
    result = evaluate.top_confusions([0, 0, 0, 1, 2], [1, 1, 2, 1, 2], names)
    assert result == [('a', 'b', 2), ('a', 'c', 1)]


def test_confusion_matrix_has_one_entry_per_class():
    """Each class gets its own 2 x 2 confusion matrix."""
    labels_dict = {'a': 0, 'b': 1}
    matrices = evaluate.get_confusion_matrix([0, 1, 1], [0, 1, 0], labels_dict, ['a', 'b'])
    assert set(matrices) == {'a', 'b'}
    assert matrices['a'].shape == (2, 2)
