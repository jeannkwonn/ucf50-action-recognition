"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. The LRCN model combines a 2D CNN backbone (e.g., ResNet) for spatial feature
extraction from individual frames with an LSTM to capture temporal dynamics across frames.
An additional fully-connected layer is used to output the final class predictions.

Classes:
    Identity: A helper module that returns the input unchanged. It replaces the fully-connected
            layer of the ResNet backbone.
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization,
            and a final fully-connected layer to produce class logits.
"""

from torch import nn
from torchvision import models

RESNET_VARIANTS = ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')

class Identity(nn.Module):
    """
    A placeholder identity operator that is argument-insensitive.

    This module is used to replace the fully-connected (fc) layer in the ResNet backbone,
    effectively making the backbone output the raw features before classification.

    Example:
        >>> identity = Identity()
        >>> output = identity(input_tensor)
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """
        Return the input as is.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: The same tensor x.
        """
        return x

class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.

    This model uses a ResNet backbone as a 2D CNN to extract spatial features from each frame.
    An LSTM then models how those features change across the sequence of frames.
    Dropout is applied before the final fully-connected layer that produces class logits.

    Args:
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): If True, uses ImageNet pretrained ResNet weights.
        cnn_model (str, optional): ResNet variant used as the backbone. One of 'resnet18',
                'resnet34', 'resnet50', 'resnet101', 'resnet152'. Default is 'resnet34'.
        temporal_pool (str, optional): How the LSTM outputs over time are combined: 'last' uses
                only the last time step, 'mean' averages all time steps. Default is 'last'.

    Raises:
        ValueError: If the specified cnn_model is not supported.
    """

    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes,
                 pretrained=True, cnn_model='resnet34', temporal_pool='last'):
        super().__init__()

        if cnn_model not in RESNET_VARIANTS:
            raise ValueError(
                f'Unsupported CNN backbone "{cnn_model}". Choose one of {RESNET_VARIANTS}.')

        if temporal_pool not in ('last', 'mean'):
            raise ValueError(f'temporal_pool must be "last" or "mean", got "{temporal_pool}".')
        self.temporal_pool = temporal_pool

        # BUG FIX: 'resnet152' used to silently load resnet34. Now every name loads
        # the matching model. "DEFAULT" selects the best available ImageNet weights.
        weights = 'DEFAULT' if pretrained else None
        base_cnn = getattr(models, cnn_model)(weights=weights)

        # Replace the original fc layer with an identity mapping so raw features come out.
        num_features = base_cnn.fc.in_features
        base_cnn.fc = Identity()
        self.base_model = base_cnn

        # BUG FIX: batch_first=True. Without it the LSTM reads (batch, time, features) as
        # (time, batch, features), which treats the videos in a batch as consecutive frames
        # of one video and mixes information between unrelated videos.
        self.rnn = nn.LSTM(num_features, hidden_size, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        Forward pass for the LRCN model.

        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)

        For each time step (frame), the CNN backbone extracts features. These features are then
        passed through the LSTM sequentially. The output from the last time step is then passed
        through dropout and the final fully-connected layer to produce the class logits.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        batch_size, time_steps, channels, height, width = x.shape

        # (B, T, C, H, W) -> (B*T, C, H, W) -> CNN -> (B*T, F) -> (B, T, F)
        frames = x.reshape(batch_size * time_steps, channels, height, width)
        features = self.base_model(frames).reshape(batch_size, time_steps, -1)

        out, _ = self.rnn(features)          # (B, T, hidden)
        if self.temporal_pool == 'mean':
            out = out.mean(dim=1)            # average over time -> (B, hidden)
        else:
            out = out[:, -1]                 # last time step -> (B, hidden)
        return self.fc(self.dropout(out))
