import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        learning_rate=0.01,
        max_epochs=100,
        batch_size=32,
        regularization=0,
        patience=3,
    ):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.patience = patience
        self.weights = None
        self.bias = None

    def softmax(self, z):
        """Compute softmax activation function."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Prevent overflow
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # Initialize weights and bias for each class
        self.weights = np.random.uniform(
            low=-1.0, high=1.0, size=(num_features, num_classes)
        )
        self.bias = np.random.uniform(low=-1.0, high=1.0, size=(1, num_classes))

        # One-hot encode the labels
        y_one_hot = np.eye(num_classes)[y]

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Shuffle training data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled, y_shuffled = X[indices], y_one_hot[indices]

            # Mini-batch Gradient Descent
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]

                # Forward pass
                logits = X_batch @ self.weights + self.bias
                y_pred = self.softmax(logits)

                # Compute gradients
                error = y_pred - y_batch
                gradient_w = (X_batch.T @ error) / X_batch.shape[
                    0
                ] + 2 * self.regularization * self.weights
                gradient_b = np.mean(error, axis=0, keepdims=True)

                # Update parameters
                self.weights -= self.learning_rate * gradient_w
                self.bias -= self.learning_rate * gradient_b

            # Compute training loss
            train_logits = X @ self.weights + self.bias
            y_train_pred = self.softmax(train_logits)
            train_loss = self.categorical_cross_entropy(y_one_hot, y_train_pred)

            # Early stopping check
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            elif self.patience > 0:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    def categorical_cross_entropy(self, y_true, y_pred):
        """Compute categorical cross-entropy loss."""
        epsilon = 1e-10
        return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))

    def predict(self, X):
        """Predict class labels."""
        logits = X @ self.weights + self.bias
        probabilities = self.softmax(logits)
        return np.argmax(
            probabilities, axis=1
        )  # Return the class with the highest probability

    def score(self, X, y):
        """Compute accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def save(self, filename):
        """Save model weights and bias."""
        np.savez(filename, weights=self.weights, bias=self.bias)

    def load(self, filename):
        """Load model weights and bias."""
        data = np.load(filename)
        self.weights = data["weights"]
        self.bias = data["bias"]
