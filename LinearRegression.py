from typing import Optional
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size: int = batch_size
        self.regularization: float = regularization
        self.max_epochs: int = max_epochs
        self.patience: int = patience

        # Optional attributes and lists to store training and validation losses
        self.weights: Optional[np.ndarray] = None  # Shape: (num_features, num_outputs)
        self.bias: Optional[np.ndarray] = None  # Shape: (1, num_outputs)
        self.training_losses: list[float] = []  # Store training loss per epoch
        self.validation_losses: list[float] = []  # Store validation loss per epoch

    def fit(
        self,
        X,
        y,
        batch_size=32,
        regularization=0,
        max_epochs=100,
        patience=3,
        noise_scale=0.1,
    ):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # 0th shape is the number of features, 1st shape is the number of outputs
        num_samples, num_features = X.shape
        num_outputs = y.shape[1]

        # Initialize the weights and bias based on the shape of X and y.
        self.weights: np.ndarray = np.random.uniform(
            low=-1.0, high=1.0, size=(num_features, num_outputs)
        )
        self.bias: np.ndarray = np.random.uniform(
            low=-1.0, high=1.0, size=(1, num_outputs)
        )

        # Some Gaussian noise of 0.1
        noise_weight: np.ndarray = np.random.normal(
            loc=0.0, scale=0.1, size=(num_features, num_outputs)
        )

        # Split into training and validation sets
        # random_state is arbitrary number
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=1
        )

        best_loss: float = float("inf")
        patience_counter: int = 0

        for epoch in range(max_epochs):
            # Shuffle data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            # Process batches
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                # Predictions
                # @ is matrix multiplication
                y_pred: np.ndarray = X_batch @ self.weights + self.bias

                # Compute gradients
                error: np.ndarray = y_pred - y_batch
                gradient_w: np.ndarray = (2 / X_batch.shape[0]) * (
                    X_batch.T @ error
                ) + 2 * regularization * self.weights
                gradient_b: np.ndarray = (2 / X_batch.shape[0]) * np.sum(
                    error, axis=0, keepdims=True
                )

                # Add Gaussian noise to weights during update
                noise_weight: np.ndarray = np.random.normal(
                    loc=0.0, scale=noise_scale, size=self.weights.shape
                )
                noise_bias: np.ndarray = np.random.normal(
                    loc=0.0, scale=noise_scale, size=self.bias.shape
                )

                # Update parameters with noise
                self.weights -= 0.01 * (
                    gradient_w + noise_weight
                )  # Learning rate = 0.01
                self.bias -= 0.01 * (gradient_b + noise_bias)

            # Compute validation loss
            y_val_pred: np.ndarray = X_val @ self.weights + self.bias
            val_loss: float = np.mean((y_val - y_val_pred) ** 2)

            # Save loss values
            self.validation_losses.append(val_loss)

            # Early stopping
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

        # After each model trains, plot the loss against the step number
        plt.plot(self.validation_losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.show()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # Prediction function.
        return X @ self.weights + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # Scoring function.
        predictions: np.ndarray = self.predict(X)

        # Mean squared error.
        mse: float = np.mean((y - predictions) ** 2)

        return mse

    # Save the weights and bias to a file
    def save(self, filename: str):
        """Save the model parameters to a file.

        Parameters
        ----------
        filename: str
            The name of the file to save the model parameters.
        """
        np.savez(
            filename,
            weights=self.weights,
            bias=self.bias,
        )

    # Load the weights and bias from a file
    def load(self, filename: str):
        """Load the model parameters from a file.

        Parameters
        ----------
        filename: str
            The name of the file to load the model parameters.
        """
        data = np.load(filename)
        self.weights = data["weights"]
        self.bias = data["bias"]
