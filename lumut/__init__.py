"""
Lumut - Lightweight Machine Learning Toolkit
=============================================

Lumut is a lightweight, modular ML toolkit designed to quietly stick 
into any Python project, like its name "lumut" (moss in Indonesian).

Main modules
------------
- base: Base classes for estimators, classifiers, regressors
- models: Machine learning models (linear, tree, etc.)
- preprocessing: Data preprocessing utilities
- metrics: Evaluation metrics
- utils: Utility functions

Example
-------
>>> from lumut.models import LinearRegression
>>> from lumut.preprocessing import StandardScaler
>>> from lumut.metrics import r2_score
>>> 
>>> # Create and fit model
>>> model = LinearRegression()
>>> model.fit(X_train, y_train)
>>> 
>>> # Make predictions
>>> y_pred = model.predict(X_test)
>>> score = r2_score(y_test, y_pred)
"""

__version__ = "0.1.0"
__author__ = "Zul"
__email__ = "zul12.project@gmail.com"

# Base classes
from .base import (
    BaseEstimator,
    BaseClassifier,
    BaseRegressor,
    BaseTransformer,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
)

# Models
from .models.linear import LinearRegression, LogisticRegression

# Preprocessing
from .preprocessing.scalers import StandardScaler, MinMaxScaler

# Metrics
from .metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
)

__all__ = [
    # Base classes
    "BaseEstimator",
    "BaseClassifier",
    "BaseRegressor",
    "BaseTransformer",
    "ClassifierMixin",
    "RegressorMixin",
    "TransformerMixin",
    # Models
    "LinearRegression",
    "LogisticRegression",
    # Preprocessing
    "StandardScaler",
    "MinMaxScaler",
    # Metrics
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "confusion_matrix",
]