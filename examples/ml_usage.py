"""
Contoh penggunaan Lumut ML Library
"""
import numpy as np
from lumut.models import LinearRegression, LogisticRegression
from lumut.preprocessing import StandardScaler, MinMaxScaler
from lumut.metrics import (
    r2_score, 
    mean_squared_error,
    accuracy_score,
    confusion_matrix
)


def example_linear_regression():
    """Contoh Linear Regression"""
    print("=" * 50)
    print("LINEAR REGRESSION EXAMPLE")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 1) * 10
    y = 3 * X.squeeze() + 7 + np.random.randn(100) * 2
    
    # Split data
    split_idx = 80
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print()


def example_logistic_regression():
    """Contoh Logistic Regression"""
    print("=" * 50)
    print("LOGISTIC REGRESSION EXAMPLE")
    print("=" * 50)
    
    # Generate synthetic binary classification data
    np.random.seed(42)
    
    # Class 0
    X_class0 = np.random.randn(50, 2) + np.array([2, 2])
    y_class0 = np.zeros(50)
    
    # Class 1
    X_class1 = np.random.randn(50, 2) + np.array([-2, -2])
    y_class1 = np.ones(50)
    
    # Combine
    X = np.vstack([X_class0, X_class1])
    y = np.concatenate([y_class0, y_class1])
    
    # Shuffle
    shuffle_idx = np.random.permutation(100)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Split data
    split_idx = 80
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"\nSample Predictions (first 5):")
    for i in range(min(5, len(y_test))):
        print(f"  True: {int(y_test[i])}, Pred: {int(y_pred[i])}, "
              f"Proba: [{y_proba[i][0]:.3f}, {y_proba[i][1]:.3f}]")
    print()


def example_preprocessing():
    """Contoh Preprocessing"""
    print("=" * 50)
    print("PREPROCESSING EXAMPLE")
    print("=" * 50)
    
    # Sample data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    
    print("Original data:")
    print(X)
    print()
    
    # StandardScaler
    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(X)
    print("After StandardScaler:")
    print(X_std)
    print(f"Mean: {std_scaler.mean_}")
    print(f"Std: {std_scaler.scale_}")
    print()
    
    # MinMaxScaler
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    X_minmax = minmax_scaler.fit_transform(X)
    print("After MinMaxScaler:")
    print(X_minmax)
    print(f"Data min: {minmax_scaler.data_min_}")
    print(f"Data max: {minmax_scaler.data_max_}")
    print()
    
    # Inverse transform
    X_recovered = std_scaler.inverse_transform(X_std)
    print("Recovered from StandardScaler:")
    print(X_recovered)
    print()


if __name__ == "__main__":
    example_preprocessing()
    example_linear_regression()
    example_logistic_regression()