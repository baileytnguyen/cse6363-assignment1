import numpy as np
from sklearn.datasets import load_iris
from LogisticRegression import LogisticRegression

def main():
    # Load the Iris dataset
    data = load_iris()
    X = data.data
    y = (data.target == 0).astype(int)  # Binary classification: class 0 vs. others

    # Select features: petal length and sepal width
    features = X[:, [2, 1]]  # Petal length (index 2), Sepal width (index 1)

    # Initialize and train the logistic regression model
    model = LogisticRegression(learning_rate=0.1, max_epochs=1000)
    model.fit(features, y)

    # Save the weights and bias to a file
    np.savez("logreg_petal_width_model.npz", weights=model.weights, bias=model.bias)

    print("Model trained and parameters saved to 'logreg_petal_width_model.npz'.")

if __name__ == "__main__":
    main()
