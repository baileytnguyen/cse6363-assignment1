from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression
import numpy as np


# test linreg_sepalwidth_petallength_target_petalwidth_model.npz
def main():
    # Load the Iris dataset
    data = load_iris()

    # Extract the features and target
    X = data["data"][:, [1, 2]]  # Sepal width (index 1), Petal length (index 2)
    y = data["data"][:, [3]]  # Petal width (index 3)

    num_bins = 3  # Try 3 or 4 based on the histogram
    y_binned = np.digitize(y, np.histogram(y, bins=num_bins)[1])

    # Split the data into training and testing sets (10% for testing, stratified by class)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y_binned
    )

    # Load model
    model = LinearRegression()
    model.load("linreg_sepalwidth_petallength_target_petalwidth_model.npz")

    # Score error using score function
    error = model.score(X_test, y_test)

    print(
        f"Accuracy of the 'linreg_sepalwidth_petallength_target_petalwidth_model.npz' model on the test set: {error:.2f}"
    )


if __name__ == "__main__":
    main()
