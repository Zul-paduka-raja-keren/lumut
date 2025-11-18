"""
Metrics for evaluating model performance.
"""
import numpy as np
from typing import Optional


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy classification score.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
        
    Returns
    -------
    score : float
        Fraction of correctly classified samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Compute precision score.
    
    Precision = TP / (TP + FP)
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    pos_label : int, default=1
        The label of the positive class.
        
    Returns
    -------
    precision : float
        Precision score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Compute recall score.
    
    Recall = TP / (TP + FN)
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    pos_label : int, default=1
        The label of the positive class.
        
    Returns
    -------
    recall : float
        Recall score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """
    Compute F1 score.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    pos_label : int, default=1
        The label of the positive class.
        
    Returns
    -------
    f1 : float
        F1 score.
    """
    precision = precision_score(y_true, y_pred, pos_label)
    recall = recall_score(y_true, y_pred, pos_label)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean squared error regression loss.
    
    MSE = mean((y_true - y_pred)^2)
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.
        
    Returns
    -------
    mse : float
        Mean squared error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean absolute error regression loss.
    
    MAE = mean(|y_true - y_pred|)
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.
        
    Returns
    -------
    mae : float
        Mean absolute error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R^2 (coefficient of determination) regression score.
    
    R^2 = 1 - (SS_res / SS_tot)
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.
        
    Returns
    -------
    r2 : float
        R^2 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
        
    Returns
    -------
    cm : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(labels)
    
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    
    return cm