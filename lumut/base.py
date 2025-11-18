"""
Base classes for Lumut ML library.
Inspired by scikit-learn's base estimator pattern.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Optional


class BaseEstimator(ABC):
    """Base class for all estimators in Lumut."""
    
    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                params[key] = getattr(self, key)
        return params
    
    def set_params(self, **params) -> 'BaseEstimator':
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ClassifierMixin:
    """Mixin class for all classifiers in Lumut."""
    
    _estimator_type = "classifier"
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
            
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class RegressorMixin:
    """Mixin class for all regressors in Lumut."""
    
    _estimator_type = "regressor"
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.
            
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        from .metrics import r2_score
        return r2_score(y, self.predict(X))


class TransformerMixin:
    """Mixin class for all transformers in Lumut."""
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values (None for unsupervised transformations).
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features_new)
            Transformed array.
        """
        return self.fit(X, y).transform(X)


class BaseClassifier(BaseEstimator, ClassifierMixin, ABC):
    """Abstract base class for classifiers."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassifier':
        """Fit the model according to the given training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        pass


class BaseRegressor(BaseEstimator, RegressorMixin, ABC):
    """Abstract base class for regressors."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseRegressor':
        """Fit the model according to the given training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples in X."""
        pass


class BaseTransformer(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for transformers."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseTransformer':
        """Fit the transformer according to the given training data."""
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        pass