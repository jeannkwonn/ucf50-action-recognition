"""
Module: evaluate.py

This module provides functions for evaluating a video classification model on test data.
It includes functions to compute predictions and accuracy, generate a detailed classification report,
and compute a multilabel confusion matrix for all classes.

Functions:
    - test: Evaluates the model on a test DataLoader and returns the ground truth labels,
      predicted labels, and overall accuracy.
    - get_test_report: Generates a classification report using scikit-learn's classification_report.
    - get_confusion_matrix: Computes a multilabel confusion matrix for each class.
    - predict_probs: Returns the class probabilities of every sample (to blend models).
    - accuracy_by_key: Computes accuracy separately for each group, for example each class.
    - top_confusions: Lists the most common (true class, predicted class) mistakes.
"""

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix

def _scores(model, x_batch, tta):
    """Run the model on a batch, optionally averaging with the mirrored batch (TTA)."""
    output = model(x_batch)
    if tta:
        output = (output + model(torch.flip(x_batch, dims=[-1]))) / 2
    return output

def predict_probs(model, dataloader, device, tta=False):
    """
    Compute the class probabilities for every sample, so predictions of models can be blended.

    Args:
        model (torch.nn.Module): The trained video classification model.
        dataloader (torch.utils.data.DataLoader): DataLoader to score (not shuffled).
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.
        tta (bool, optional): If True, average each clip with its mirrored copy. Default is False.

    Returns:
        tuple: (targets, probs)
            - targets (list): Ground truth labels for all samples.
            - probs (numpy.ndarray): Array of shape (n_samples, n_classes) with probabilities.
    """
    model.eval()
    targets, probs = [], []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader):
            output = _scores(model, x_batch.to(device), tta)
            probs.append(torch.softmax(output, dim=1).cpu())
            targets.extend(y_batch.tolist())
    return targets, torch.cat(probs).numpy()

def test(model, dataloader, device, tta=False):
    """
    Evaluate the model on the test dataset and compute overall accuracy.

    This function sets the model to evaluation mode and processes the test data
    from the provided DataLoader. It computes predictions for each batch, counts the number
    of correct predictions, and accumulates the true and predicted labels.

    Args:
        model (torch.nn.Module): The trained video classification model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.
        tta (bool, optional): If True, use test time augmentation. Each clip is also shown
            mirrored left to right, and the two sets of scores are averaged. Default is False.

    Returns:
        tuple: (targets, outputs, accuracy)
            - targets (list): Ground truth labels for all samples.
            - outputs (list): Predicted labels for all samples.
            - accuracy (float): Overall accuracy computed as the ratio of correct predictions
                                to the total number of samples.
    """
    model.eval()
    with torch.no_grad():
        total_correct_preds = 0.0
        len_dataset = len(dataloader.dataset)
        targets, outputs = [], []
        for x_batch, y_batch in tqdm(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = _scores(model, x_batch, tta)
            pred = output.argmax(dim=1, keepdim=True)
            correct_preds = pred.eq(y_batch.view_as(pred)).sum().item()
            total_correct_preds += correct_preds
            outputs.extend(pred.view(-1).detach().cpu().numpy().tolist())
            targets.extend(y_batch.detach().cpu().numpy().tolist())

        accuracy = total_correct_preds / float(len_dataset)

    return targets, outputs, accuracy

def accuracy_by_key(keys, targets, outputs):
    """
    Compute accuracy separately for each key, for example for each class or each UCF50 group.

    Args:
        keys (list): One key per sample (a class name or a group number).
        targets (list): Ground truth labels, in the same order as keys.
        outputs (list): Predicted labels, in the same order as keys.

    Returns:
        dict: Maps each key to the fraction of its samples that were predicted correctly.
    """
    totals, correct = {}, {}
    for key, target, output in zip(keys, targets, outputs):
        totals[key] = totals.get(key, 0) + 1
        correct[key] = correct.get(key, 0) + int(target == output)
    return {key: correct[key] / count for key, count in totals.items()}

def top_confusions(targets, outputs, names, top_k=15):
    """
    Find the most common mistakes: pairs of (true class, wrongly predicted class).

    Args:
        targets (list): Ground truth numeric labels.
        outputs (list): Predicted numeric labels.
        names (dict): Maps a numeric label to its class name.
        top_k (int, optional): How many pairs to return. Default is 15.

    Returns:
        list: Tuples (true_name, predicted_name, count), most frequent first.
    """
    counts = {}
    for target, output in zip(targets, outputs):
        if target != output:
            pair = (names[target], names[output])
            counts[pair] = counts.get(pair, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [(true, pred, count) for (true, pred), count in ranked[:top_k]]

def get_test_report(target, output, target_names):
    """
    Generate a detailed classification report based on test results.

    This function uses scikit-learn's classification_report to produce a dictionary
    containing precision, recall, F1-score, and support for each class.

    Args:
        target (list): Ground truth labels.
        output (list): Predicted labels.
        target_names (list): List of class names corresponding to the labels.

    Returns:
        dict: A classification report as a dictionary.
    """
    return classification_report(target, output, output_dict=True, target_names=target_names)

def get_confusion_matrix(targets, outputs, labels_dict, all_cats):
    """
    Compute a multilabel confusion matrix for each class.

    This function converts numeric labels to their corresponding class names using the provided
    labels_dict, then computes a multilabel confusion matrix for each class using scikit-learn.

    Args:
        targets (list): Ground truth numeric labels.
        outputs (list): Predicted numeric labels.
        labels_dict (dict): Dictionary mapping class names to numeric labels.
        all_cats (list): List of all class names.

    Returns:
        dict: A dictionary where keys are class names and values are the corresponding confusion matrices.
    """
    # Create an inverse mapping from numeric label to class name
    inv_labels_dict = {label: cat for cat, label in labels_dict.items()}
    target_cats = [inv_labels_dict[target] for target in targets]
    output_cats = [inv_labels_dict[output] for output in outputs]
    confusion_mat = multilabel_confusion_matrix(target_cats, output_cats, labels=all_cats)
    return dict(zip(all_cats, confusion_mat))
