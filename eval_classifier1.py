import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

def main():
    # Load the Iris dataset
    data = load_iris()
    X = data.data
    y = (data.target == 0).astype(int)  # Binary classification: class 0 vs. others

    # Use petal length and sepal width to predict petal width
    features = X[:, [2, 1]]  # Petal length (index 2), Sepal width (index 1)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

    # Load the saved model parameters
    model_data = np.load("logreg_petal_width_model.npz")
    weights = model_data["weights"]
    bias = model_data["bias"]

    # Initialize the logistic regression model and set weights
    model = LogisticRegression()
    model.weights = weights
    model.bias = bias

    # Predict labels for the test set
    y_pred = model.predict(X_test)

    # Compute accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy of the model on the test set: {accuracy:.2f}")

if __name__ == "__main__":
    main()
