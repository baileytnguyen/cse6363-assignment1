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

    # Initialize and train the logistic regression model
    model = LogisticRegression(learning_rate=0.1, max_epochs=1000)
    model.fit(X_train, y_train)

    # Save the weights and bias
    np.savez("logreg_petal_width_model.npz", weights=model.weights, bias=model.bias)

    print(f"Model trained to predict petal width using petal length and sepal width.")

if __name__ == "__main__":
    main()
