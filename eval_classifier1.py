import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression


# Classification with petal length and width
def main():
    # Load the Iris dataset
    data = load_iris()

    # Extract the features and target
    X = data["data"][:, [2, 3]]  # Petal length (index 2), Petal width (index 3)
    # get species target
    y = data["target"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Train Logistic Regression
    model = LogisticRegression(
        learning_rate=0.1,
        max_epochs=500,
        regularization=0.00,
        patience=0,
    )
    model.fit(X_train, y_train)

    # Plot decision boundary
    model.plot_decision_boundary(X_train, y_train)


if __name__ == "__main__":
    main()
