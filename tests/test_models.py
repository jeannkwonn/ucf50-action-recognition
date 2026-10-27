"""Unit tests for models.py: output shape, input checks, and the batch-mixing bug fix."""

import pytest
import torch

from models import LRCN


def make_model(**kwargs):
    """Build a small, fast model without downloading pretrained weights."""
    settings = {'hidden_size': 16, 'n_layers': 1, 'dropout_rate': 0.1, 'n_classes': 5,
                'pretrained': False, 'cnn_model': 'resnet18'}
    settings.update(kwargs)
    return LRCN(**settings)


@pytest.mark.parametrize('temporal_pool', ['last', 'mean'])
def test_output_shape(temporal_pool):
    """The model returns one score per class for every video in the batch."""
    model = make_model(temporal_pool=temporal_pool).eval()
    out = model(torch.randn(3, 4, 3, 32, 32))
    assert out.shape == (3, 5)


def test_videos_in_a_batch_do_not_affect_each_other():
    """The main bug fix: a video's result must be the same alone or inside a batch."""
    torch.manual_seed(0)
    model = make_model().eval()
    videos = torch.randn(3, 4, 3, 32, 32)
    with torch.no_grad():
        together = model(videos)
        alone = model(videos[:1])
    assert torch.allclose(together[:1], alone, atol=1e-5)


def test_mean_pooling_differs_from_last_step():
    """The two pooling options are really different computations."""
    torch.manual_seed(0)
    last = make_model(temporal_pool='last').eval()
    mean = make_model(temporal_pool='mean').eval()
    mean.load_state_dict(last.state_dict())
    videos = torch.randn(2, 4, 3, 32, 32)
    with torch.no_grad():
        assert not torch.allclose(last(videos), mean(videos))


def test_unknown_backbone_is_rejected():
    """A backbone name that is not a supported ResNet raises an error."""
    with pytest.raises(ValueError):
        make_model(cnn_model='vgg16')


def test_unknown_temporal_pool_is_rejected():
    """Only 'last' and 'mean' are valid pooling options."""
    with pytest.raises(ValueError):
        make_model(temporal_pool='max')
