import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression


# Predict sepal width given petal length and sepal length
def main():
    # Load the Iris dataset
    data = load_iris()

    # Extract the features and target
    X = data["data"][:, [0, 2]]  # Petal length (index 2), Sepal length (index 0)
    y = data["data"][:, [1]]  # Sepal width (index 1)

    # Split the data into training and testing sets (10% for testing, stratified by class)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Initialize and train the linear regression model
    model = LinearRegression(max_epochs=100)
    model.fit(X_train, y_train, patience=0)

    # Save the weights and bias to a file
    model.save("linreg_sepallength_petallength_target_sepalwidth_model.npz")

    print(
        "Model trained and parameters saved to 'linreg_sepallength_petallength_target_sepalwidth_model.npz'."
    )


if __name__ == "__main__":
    main()
