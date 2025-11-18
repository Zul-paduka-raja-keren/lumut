"""
Preprocessing utilities for Lumut.
Data scaling and normalization tools.
"""
import numpy as np
from typing import Optional
from ..base import BaseTransformer


class StandardScaler(BaseTransformer):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    The standard score of a sample x is calculated as:
        z = (x - u) / s
    where u is the mean and s is the standard deviation.
    
    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.
        
    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        The mean value for each feature in the training set.
    scale_ : ndarray of shape (n_features,)
        The standard deviation for each feature in the training set.
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'StandardScaler':
        """
        Compute the mean and std to be used for later scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Fitted scaler.
        """
        X = np.asarray(X)
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        
        if self.with_std:
            self.scale_ = np.std(X, axis=0)
            # Avoid division by zero
            self.scale_[self.scale_ == 0] = 1.0
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        X = np.asarray(X).copy()
        
        if self.with_mean:
            X -= self.mean_
        
        if self.with_std:
            X /= self.scale_
        
        return X
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale back the data to the original representation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to inverse transform.
            
        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Original data.
        """
        X = np.asarray(X).copy()
        
        if self.with_std:
            X *= self.scale_
        
        if self.with_mean:
            X += self.mean_
        
        return X


class MinMaxScaler(BaseTransformer):
    """
    Transform features by scaling each feature to a given range.
    
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.
    
    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
        
    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the training data.
    max_ : ndarray of shape (n_features,)
        Per feature maximum seen in the training data.
    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data.
    """
    
    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'MinMaxScaler':
        """
        Compute the minimum and maximum to be used for later scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Fitted scaler.
        """
        X = np.asarray(X)
        
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        
        data_range = self.data_max_ - self.data_min_
        # Avoid division by zero
        data_range[data_range == 0] = 1.0
        
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features of X according to feature_range.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        X = np.asarray(X)
        X_scaled = X * self.scale_ + self.min_
        return X_scaled
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Undo the scaling of X according to feature_range.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to inverse transform.
            
        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Original data.
        """
        X = np.asarray(X)
        X_original = (X - self.min_) / self.scale_
        return X_original