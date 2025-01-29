from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression


# test linreg_sepallength_target_petallength_model.npz
def main():
    # Load the Iris dataset
    data = load_iris()

    # Extract the features and target
    X = data["data"][:, [0]]  # Sepal length (index 0)
    y = data["data"][:, [2]]  # Petal length (index 2)

    # Split the data into training and testing sets (10% for testing, stratified by class)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Load model
    model = LinearRegression()
    model.load("linreg_sepallength_target_petallength_model.npz")

    # Score mse using score function
    error = model.score(X_test, y_test)

    print(
        f"Mean squared error of the 'linreg_sepallength_target_petallength_model.npz' model on the test set: {error:.2f}"
    )


if __name__ == "__main__":
    main()
