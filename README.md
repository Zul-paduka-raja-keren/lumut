# 🌿 Lumut - Machine Learning Toolkit

**Lumut** is a lightweight, modular machine learning toolkit designed to quietly stick into any Python project, like its name "lumut" (Indonesian word for moss). Inspired by scikit-learn, Lumut provides simple and efficient tools for machine learning with a clean, intuitive API.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/numpy-required-orange.svg)](https://numpy.org/)

---

## 🚀 Features

- **🎯 Supervised Learning**: Linear & Logistic Regression
- **🔧 Preprocessing**: StandardScaler, MinMaxScaler
- **📊 Metrics**: Accuracy, Precision, Recall, F1, MSE, MAE, R², Confusion Matrix
- **🏗️ Modular Design**: Use only what you need
- **🐍 Pure Python**: Easy to understand and extend
- **📝 Type Hints**: Full type annotation support
- **🧪 Well-Tested**: Comprehensive test coverage

---

## 📦 Installation

### From PyPI (when published)
```bash
pip install lumut
```

### From Source
```bash
git clone https://github.com/Zul-paduka-raja-keren/lumut.git
cd lumut
pip install -e .
```

### Requirements
- Python >= 3.8
- NumPy >= 1.20.0

---

## 🎯 Quick Start

### Linear Regression Example

```python
import numpy as np
from lumut.models import LinearRegression
from lumut.preprocessing import StandardScaler
from lumut.metrics import r2_score, mean_squared_error

# Generate sample data
X = np.random.randn(100, 1) * 10
y = 3 * X.squeeze() + 7 + np.random.randn(100) * 2

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Predict
y_pred = model.predict(X_scaled)

# Evaluate
print(f"R² Score: {r2_score(y, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y, y_pred):.4f}")
```

### Logistic Regression Example

```python
import numpy as np
from lumut.models import LogisticRegression
from lumut.preprocessing import StandardScaler
from lumut.metrics import accuracy_score, confusion_matrix

# Generate binary classification data
np.random.seed(42)
X_class0 = np.random.randn(50, 2) + [2, 2]
X_class1 = np.random.randn(50, 2) + [-2, -2]
X = np.vstack([X_class0, X_class1])
y = np.array([0] * 50 + [1] * 50)

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(learning_rate=0.1, max_iter=1000)
model.fit(X_scaled, y)

# Predict
y_pred = model.predict(X_scaled)

# Evaluate
print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")
```

---

## 📚 API Reference

### Models

#### `LinearRegression`
Ordinary least squares linear regression.

**Parameters:**
- `fit_intercept` (bool): Whether to calculate the intercept (default: True)

**Methods:**
- `fit(X, y)`: Fit the model
- `predict(X)`: Make predictions

#### `LogisticRegression`
Binary classification using logistic regression.

**Parameters:**
- `learning_rate` (float): Learning rate for gradient descent (default: 0.01)
- `max_iter` (int): Maximum iterations (default: 1000)
- `tol` (float): Tolerance for convergence (default: 1e-4)

**Methods:**
- `fit(X, y)`: Fit the model
- `predict(X)`: Predict class labels
- `predict_proba(X)`: Predict class probabilities

### Preprocessing

#### `StandardScaler`
Standardize features by removing mean and scaling to unit variance.

**Parameters:**
- `with_mean` (bool): Center the data (default: True)
- `with_std` (bool): Scale to unit variance (default: True)

**Methods:**
- `fit(X)`: Compute mean and std
- `transform(X)`: Standardize the data
- `fit_transform(X)`: Fit and transform in one step
- `inverse_transform(X)`: Scale back to original

#### `MinMaxScaler`
Scale features to a given range.

**Parameters:**
- `feature_range` (tuple): Desired range (default: (0, 1))

**Methods:**
- `fit(X)`: Compute min and max
- `transform(X)`: Scale the data
- `fit_transform(X)`: Fit and transform in one step
- `inverse_transform(X)`: Scale back to original

### Metrics

#### Classification Metrics
- `accuracy_score(y_true, y_pred)`: Accuracy
- `precision_score(y_true, y_pred)`: Precision
- `recall_score(y_true, y_pred)`: Recall
- `f1_score(y_true, y_pred)`: F1 score
- `confusion_matrix(y_true, y_pred)`: Confusion matrix

#### Regression Metrics
- `mean_squared_error(y_true, y_pred)`: MSE
- `mean_absolute_error(y_true, y_pred)`: MAE
- `r2_score(y_true, y_pred)`: R² coefficient

---

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/Zul-paduka-raja-keren/lumut.git
cd lumut

# Install with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lumut --cov-report=html

# Run specific test file
pytest tests/test_core.py
```

### Code Quality

```bash
# Format code
black lumut/

# Lint code
flake8 lumut/

# Type checking
mypy lumut/
```

---

## 🗺️ Roadmap

### v0.2.0 (Planned)
- [ ] Decision Trees
- [ ] K-Nearest Neighbors
- [ ] Train/Test Split utility
- [ ] Cross-validation

### v0.3.0 (Planned)
- [ ] Random Forest
- [ ] Gradient Boosting
- [ ] Feature selection tools
- [ ] Pipeline support

### Future
- [ ] Neural Networks (basic)
- [ ] Clustering algorithms
- [ ] Dimensionality reduction
- [ ] More preprocessing tools

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

Please ensure:
- Code follows the existing style
- All tests pass
- New features have tests
- Documentation is updated

---

## 📖 Examples

Check out the `examples/` directory for more detailed examples:
- `ml_usage.py`: Complete examples with all features
- More examples coming soon!

---

## 🐛 Bug Reports

Found a bug? Please open an issue on [GitHub Issues](https://github.com/Zul-paduka-raja-keren/lumut/issues) with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Zul**
- Email: zul12.project@gmail.com
- GitHub: [@Zul-paduka-raja-keren](https://github.com/Zul-paduka-raja-keren)

---

## 🙏 Acknowledgments

- Inspired by [scikit-learn](https://scikit-learn.org/)

---

## ⭐ Star History

If you find Lumut useful, please consider giving it a star on GitHub!

---


**Made by Zul**
