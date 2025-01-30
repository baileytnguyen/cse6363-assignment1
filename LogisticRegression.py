import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions


class LogisticRegression:
    def __init__(
        self,
        learning_rate=0.01,
        max_epochs=100,
        batch_size=32,
        regularization=0,
        patience=3,
    ):
        """Logistic Regression using Gradient Descent with Early Stopping.

        Parameters:
        -----------
        learning_rate: float
            Learning rate for gradient descent.
        max_epochs: int
            Maximum number of epochs.
        batch_size: int
            Number of samples per batch.
        regularization: float
            Regularization parameter (L2 penalty).
        patience: int
            Number of epochs to wait before stopping if validation loss does not improve.
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.patience = patience

        self.weights = None
        self.bias = None
        self.training_losses = []
        self.validation_losses = []

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, y_true, y_pred):
        """Compute binary cross-entropy loss."""
        epsilon = 1e-10  # To avoid log(0)
        return -np.mean(
            y_true * np.log(y_pred + epsilon)
            + (1 - y_true) * np.log(1 - y_pred + epsilon)
        )

    def fit(self, X, y, validation_split=0.1):
        """Train the logistic regression model using gradient descent."""
        num_samples, num_features = X.shape
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=(num_features, 1))
        self.bias = np.random.uniform(low=-1.0, high=1.0, size=(1,))

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            # Mini-batch Gradient Descent
            for i in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_train[i : i + self.batch_size]
                y_batch = y_train[i : i + self.batch_size].reshape(-1, 1)

                # Forward pass
                logits = X_batch @ self.weights + self.bias
                y_pred = self.sigmoid(logits)

                # Compute loss
                loss = self.binary_cross_entropy(y_batch, y_pred)

                # Compute gradients
                error = y_pred - y_batch
                grad_w = (X_batch.T @ error) / X_batch.shape[
                    0
                ] + 2 * self.regularization * self.weights
                grad_b = np.mean(error, axis=0)

                # Parameter update
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            # Compute validation loss
            val_logits = X_val @ self.weights + self.bias
            y_val_pred = self.sigmoid(val_logits)
            val_loss = self.binary_cross_entropy(y_val.reshape(-1, 1), y_val_pred)

            self.validation_losses.append(val_loss)

            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()
            elif self.patience > 0:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best weights
        self.weights = best_weights
        self.bias = best_bias

        # Plot validation loss
        plt.plot(self.validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss Over Epochs")
        plt.legend()
        plt.show()

    def predict(self, X):
        """Predict class labels (0 or 1)."""
        probabilities = self.sigmoid(X @ self.weights + self.bias)
        return (probabilities >= 0.5).astype(int)

    def plot_decision_boundary(self, X, y):
        """Plot decision regions for 2D datasets."""
        if X.shape[1] != 2:
            print("Decision boundary plotting only supports 2D features.")
            return

        y_flat = y.ravel()
        plot_decision_regions(X, y_flat, clf=self)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Decision Boundary")
        plt.show()

    def predict_proba(self, X):
        """Return probability estimates."""
        return self.sigmoid(X @ self.weights + self.bias)

    def __call__(self, X):
        """For compatibility with mlxtend plot_decision_regions."""
        return self.predict(X)
