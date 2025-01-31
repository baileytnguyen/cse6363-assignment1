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
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # load Logistic Regression
    model = LogisticRegression()
    model.load("logreg_petalwidth_petalwidth_target_species_model.npz")

    # Print model accuracy
    print("Model accuracy: ", model.score(X_test, y_test))

    # Plot decision boundary
    # model.plot_decision_boundary(X_train, y_train)


if __name__ == "__main__":
    main()
