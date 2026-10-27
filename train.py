"""
Module: train.py

This module provides functions to train and evaluate a video classification model.
It includes the main training loop that tracks loss and accuracy over epochs,
updates the best performing model based on validation accuracy, and provides helper
functions to compute the learning rate, batch loss, and epoch loss.

Functions:
    - train: Runs the training and validation loops over a specified number of epochs.
    - get_learning_rate: Retrieves the current learning rate from an optimizer.
    - batch_correct_preds: Computes the number of correct predictions in a mini-batch.
    - get_batch_loss: Computes the loss for a mini-batch and performs backpropagation.
    - get_epoch_loss: Computes the average loss and accuracy over an epoch.
"""

import os
import copy
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# pylint: disable=too-many-locals
def train(dataloaders, model, criterion, optimizer, scheduler, device, optim_model_wts_dir, n_epochs=30, log_fn=None,
          eval_criterion=None):
    """
    Train and validate the model over a given number of epochs.

    This function performs the training loop for a video classification model.
    It iterates through the specified number of epochs, updates model weights
    using backpropagation, and evaluates model performance on a validation set.
    The model with the best validation accuracy is saved to disk.

    Args:
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders. An optional 'test'
            DataLoader is evaluated every epoch for logging only; it is never used to pick weights
            or the learning rate.
        model (torch.nn.Module): The video classification model to be trained.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler. ReduceLROnPlateau uses the
            validation loss; other schedulers just step once per epoch.
        device (torch.device): Device (CPU or GPU) on which to perform training.
        optim_model_wts_dir (str): Directory to save the best model weights.
        n_epochs (int, optional): Number of training epochs. Default is 30.
        log_fn (callable, optional): Called as log_fn(metrics, epoch) after every epoch, where
            metrics is a dict of that epoch's numbers. Used to log to Weights & Biases.
        eval_criterion (callable, optional): Loss used for the validation and test numbers.
            Defaults to criterion. Use a plain loss here when criterion uses label smoothing, so
            the reported validation/test losses stay comparable between runs.

    Returns:
        tuple: (model, loss_hist, acc_hist)
            - model (torch.nn.Module): The trained model loaded with the best validation weights.
            - loss_hist (dict): Lists of training, validation (and test, if given) losses per epoch.
            - acc_hist (dict): Lists of training, validation (and test, if given) accuracies per epoch.
    """
    eval_criterion = eval_criterion if eval_criterion is not None else criterion
    loss_hist = {'train': [], 'val': [], 'test': []}
    acc_hist = {'train': [], 'val': [], 'test': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(n_epochs):
        current_lr = get_learning_rate(optimizer)
        print(f'Epoch {epoch + 1}/{n_epochs}; Current learning rate {current_lr}')

        # Training phase
        model.train()
        train_loss, train_accuracy = get_epoch_loss(model, criterion, dataloaders['train'],
                                                    device, optimizer)
        loss_hist['train'].append(train_loss)
        acc_hist['train'].append(train_accuracy)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = get_epoch_loss(model, eval_criterion, dataloaders['val'], device)

        # Optional test evaluation, for logging only (never used to pick weights or learning rate)
        test_loss, test_accuracy = None, None
        if 'test' in dataloaders:
            with torch.no_grad():
                test_loss, test_accuracy = get_epoch_loss(model, eval_criterion,
                                                          dataloaders['test'], device)
            loss_hist['test'].append(test_loss)
            acc_hist['test'].append(test_accuracy)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            best_model_name = 'best_model_wts.pt'
            best_model_path = os.path.join(optim_model_wts_dir, best_model_name)
            torch.save(best_model_wts, best_model_path)
            print(f'Best model weights are updated at epoch {epoch + 1}!')
        loss_hist['val'].append(val_loss)
        acc_hist['val'].append(val_accuracy)

        # Update the learning rate. ReduceLROnPlateau watches the validation loss and reloads the
        # best weights when it lowers the rate; other schedulers (e.g., cosine) step each epoch.
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
            if current_lr != get_learning_rate(optimizer):
                print('Loading best model weights!')
                model.load_state_dict(best_model_wts)
        else:
            scheduler.step()

        print(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, "
              f"accuracy: {100 * val_accuracy:.2f}")
        if test_loss is not None:
            print(f"test loss: {test_loss:.6f}, test accuracy: {100 * test_accuracy:.2f}")

        # Send this epoch's numbers to the logger (e.g., Weights & Biases)
        if log_fn is not None:
            metrics = {'epoch': epoch + 1, 'learning_rate': current_lr,
                       'train/loss': train_loss, 'train/acc': train_accuracy,
                       'val/loss': val_loss, 'val/acc': val_accuracy}
            if test_loss is not None:
                metrics.update({'test/loss': test_loss, 'test/acc': test_accuracy})
            log_fn(metrics, epoch + 1)
        print("-" * 60)
        print()

    # Load the best model weights before returning the model
    model.load_state_dict(best_model_wts)
    return model, loss_hist, acc_hist

def get_learning_rate(optimizer):
    """
    Retrieve the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer from which to get the learning rate.

    Returns:
        float: The current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def batch_correct_preds(output, target):
    """
    Compute the number of correct predictions for a mini-batch.

    Args:
        output (torch.Tensor): Model outputs (logits) with shape (batch_size, num_classes).
        target (torch.Tensor): True labels with shape (batch_size).

    Returns:
        int: Number of correct predictions in the mini-batch.
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct_preds = pred.eq(target.view_as(pred)).sum().item()
    return correct_preds

def get_batch_loss(criterion, output, target, optimizer=None):
    """
    Compute the loss for a mini-batch and perform backpropagation (if optimizer is provided).

    Args:
        criterion (callable): Loss function.
        output (torch.Tensor): Model outputs for the mini-batch.
        target (torch.Tensor): True labels for the mini-batch.
        optimizer (torch.optim.Optimizer, optional): Optimizer to update model weights. If None,
            no backpropagation is performed.

    Returns:
        tuple: (loss_value, n_batch_correct_preds)
            - loss_value (float): Loss value for the mini-batch.
            - n_batch_correct_preds (int): Number of correct predictions in the mini-batch.
    """
    loss = criterion(output, target)
    with torch.no_grad():
        n_batch_correct_preds = batch_correct_preds(output, target)
    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), n_batch_correct_preds

def get_epoch_loss(model, criterion, dataloader, device, optimizer=None):
    """
    Compute the average loss and overall accuracy for an epoch.

    Iterates over the entire DataLoader, computes loss and accuracy for each mini-batch,
    and aggregates the results over the epoch.

    Args:
        model (torch.nn.Module): The video classification model.
        criterion (callable): Loss function.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device (CPU or GPU) on which to perform computations.
        optimizer (torch.optim.Optimizer, optional): If provided, used to update model weights
            during training.

    Returns:
        tuple: (loss, accuracy)
            - loss (float): Average loss over the epoch.
            - accuracy (float): Overall accuracy over the epoch.
    """
    running_loss, running_total_correct_preds = 0.0, 0.0
    len_dataset = len(dataloader.dataset)

    for x_batch, y_batch in tqdm(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch)
        batch_loss, n_batch_correct_preds = get_batch_loss(criterion, output, y_batch, optimizer)

        # The loss is a per-video average. Weight it by the batch size before summing.
        running_loss += batch_loss * x_batch.size(0)
        running_total_correct_preds += n_batch_correct_preds

    loss = running_loss / float(len_dataset)
    accuracy = running_total_correct_preds / float(len_dataset)
    return loss, accuracy
