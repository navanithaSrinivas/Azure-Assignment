import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score


class MyLinearModel:

    def __init__(self, learning_rate=0.005, iterations=1200):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def train(self, features, target):

        rows, cols = features.shape

        self.w = np.zeros(cols)
        self.b = 0

        self.error_history = []

        for _ in range(self.iterations):

            prediction = np.dot(features, self.w) + self.b

            weight_gradient = (2 / rows) * np.dot(
                features.T,
                prediction - target
            )

            bias_gradient = (2 / rows) * np.sum(
                prediction - target
            )

            self.w -= self.learning_rate * weight_gradient
            self.b -= self.learning_rate * bias_gradient

            mse = np.mean((target - prediction) ** 2)

            self.error_history.append(mse)

    def predict(self, features):
        return np.dot(features, self.w) + self.b


class MyLogisticModel:

    def __init__(self, learning_rate=0.05, iterations=1500):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def activation(self, value):
        return 1 / (1 + np.exp(-value))

    def train(self, features, labels):

        rows, cols = features.shape

        self.w = np.zeros(cols)
        self.b = 0

        for _ in range(self.iterations):

            z = np.dot(features, self.w) + self.b

            prediction = self.activation(z)

            dw = (1 / rows) * np.dot(
                features.T,
                prediction - labels
            )

            db = (1 / rows) * np.sum(
                prediction - labels
            )

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, features):

        z = np.dot(features, self.w) + self.b

        probability = self.activation(z)

        return np.where(probability >= 0.5, 1, 0)


print("\n----------------------------")
print("LINEAR REGRESSION MODEL")
print("----------------------------")

X_reg, y_reg = make_regression(
    n_samples=250,
    n_features=1,
    noise=15,
    random_state=10
)

x_train, x_test, y_train, y_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.25,
    random_state=10
)

linear_model = MyLinearModel()

linear_model.train(x_train, y_train)

predicted_values = linear_model.predict(x_test)

mse_score = mean_squared_error(
    y_test,
    predicted_values
)

print("Mean Squared Error =", mse_score)

plt.figure(figsize=(8, 5))

plt.scatter(
    x_test,
    y_test,
    label="Actual Data"
)

plt.scatter(
    x_test,
    predicted_values,
    label="Predicted Data"
)

plt.title("Linear Regression Output")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))

plt.plot(linear_model.error_history)

plt.title("Training Error")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()


print("\n----------------------------")
print("LOGISTIC REGRESSION MODEL")
print("----------------------------")

X_cls, y_cls = make_classification(
    n_samples=350,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=10
)

x_train, x_test, y_train, y_test = train_test_split(
    X_cls,
    y_cls,
    test_size=0.25,
    random_state=10
)

classifier = MyLogisticModel()

classifier.train(x_train, y_train)

result = classifier.predict(x_test)

acc = accuracy_score(
    y_test,
    result
)

print("Classification Accuracy =", acc)

plt.figure(figsize=(8, 5))

plt.scatter(
    x_test[:, 0],
    x_test[:, 1],
    c=result,
    cmap="viridis"
)

plt.title("Logistic Regression Classification")
plt.show()