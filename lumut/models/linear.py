"""
Linear models for classification and regression.
"""
import numpy as np
from typing import Optional
from ..base import BaseClassifier, BaseRegressor


class LinearRegression(BaseRegressor):
    """
    Ordinary least squares Linear Regression.
    
    Fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed
    targets in the dataset, and the targets predicted by the
    linear approximation.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Independent term in the linear model.
    """
    
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.fit_intercept:
            # Add intercept column
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            # Solve normal equation: (X'X)^-1 X'y
            params = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            self.intercept_ = params[0]
            self.coef_ = params[1:]
        else:
            # No intercept
            self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values.
        """
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_


class LogisticRegression(BaseClassifier):
    """
    Logistic Regression classifier.
    
    Uses gradient descent to find the optimal weights.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    max_iter : int, default=1000
        Maximum number of iterations for gradient descent.
    tol : float, default=1e-4
        Tolerance for stopping criterion.
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficient of the features.
    intercept_ : float
        Intercept (bias) term.
    classes_ : ndarray of shape (n_classes,)
        Class labels.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0
        self.classes_ = None
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit the logistic regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (binary: 0 or 1).
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        
        # Gradient descent
        for iteration in range(self.max_iter):
            # Forward pass
            z = X @ self.coef_ + self.intercept_
            predictions = self._sigmoid(z)
            
            # Compute gradients
            error = predictions - y
            grad_coef = (1 / n_samples) * (X.T @ error)
            grad_intercept = (1 / n_samples) * np.sum(error)
            
            # Update parameters
            self.coef_ -= self.learning_rate * grad_coef
            self.intercept_ -= self.learning_rate * grad_intercept
            
            # Check convergence
            if np.linalg.norm(grad_coef) < self.tol:
                break
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Probability estimates.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Returns the probability of the sample for each class.
        """
        X = np.asarray(X)
        z = X @ self.coef_ + self.intercept_
        prob_class_1 = self._sigmoid(z)
        prob_class_0 = 1 - prob_class_1
        return np.column_stack([prob_class_0, prob_class_1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]