import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def main():
    # Load the dataset
    data = np.loadtxt("spambase/spambase.data", delimiter=",")

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.25)

    # Build the SVM model
    model = SVC(kernel="rbf")
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print("Accuracy:", score)

if __name__ == "__main__":
    main()