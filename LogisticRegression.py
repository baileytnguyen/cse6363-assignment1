import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 1000, tolerance: float = 1e-6):
        """Logistic Regression using Gradient Descent.

        Parameters:
        -----------
        learning_rate: float
            The learning rate for gradient descent.
        max_epochs: int
            The maximum number of iterations.
        tolerance: float
            The stopping criteria for weight updates.
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the logistic regression model to the training data.

        Parameters:
        -----------
        X: np.ndarray
            The input data of shape (n_samples, n_features).
        y: np.ndarray
            The target values of shape (n_samples,).
        """
        num_samples, num_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient Descent
        for epoch in range(self.max_epochs):
            # Compute linear combination (z = XW + b)
            z = np.dot(X, self.weights) + self.bias

            # Sigmoid function for probabilities
            y_pred = 1 / (1 + np.exp(-z))

            # Compute gradients
            error = y_pred - y
            gradient_w = (1 / num_samples) * np.dot(X.T, error)
            gradient_b = (1 / num_samples) * np.sum(error)

            # Update weights and bias
            self.weights -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b

            # Stopping condition: if weight updates are very small
            if np.linalg.norm(gradient_w) < self.tolerance and abs(gradient_b) < self.tolerance:
                print(f"Converged at epoch {epoch + 1}")
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input data.

        Parameters:
        -----------
        X: np.ndarray
            The input data of shape (n_samples, n_features).

        Returns:
        --------
        np.ndarray
            Predicted class labels (0 or 1) of shape (n_samples,).
        """
        z = np.dot(X, self.weights) + self.bias
        probabilities = 1 / (1 + np.exp(-z))
        return (probabilities >= 0.5).astype(int)
