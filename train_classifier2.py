from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression


# Classification with sepal length and sepal width
def main():
    # Load the Iris dataset
    data = load_iris()

    # Extract the features and target
    X = data["data"][:, [0, 1]]  # Sepal length (index 0), Sepal width (index 1)

    # get species target
    y = data["target"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Train logistic regression
    model = LogisticRegression(
        learning_rate=0.01,
        max_epochs=500,
        regularization=0.00,
        patience=3,
    )
    model.fit(X_train, y_train)

    # Save
    model.save("logreg_sepallength_sepalwidth_target_species_model.npz")

    print(
        "Model trained and parameters saved to 'logreg_sepallength_sepalwidth_target_species_model.npz'."
    )


if __name__ == "__main__":
    main()
