from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


# Function to plot decision boundaries
def plot_decision_boundary(X, y, classifier, title):
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X, y, clf=classifier, legend=2)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.show()


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

    # Train logistic regression
    model = LogisticRegression(
        learning_rate=0.1,
        max_epochs=1000,
        regularization=0.00,
        patience=0,
    )
    model.fit(X_train, y_train)

    plot_decision_boundary(
        X_train, y_train, model, "Decision Boundary (Petal Features)"
    )

    # Save
    model.save("logreg_petalwidth_petalwidth_target_species_model.npz")

    print(
        "Model trained and parameters saved to 'logreg_petalwidth_petalwidth_target_species_model.npz'."
    )


if __name__ == "__main__":
    main()
