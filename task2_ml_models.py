import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


print("\n================================")
print("DECISION TREE CLASSIFIER")
print("================================")

wine = load_wine()

features = wine.data
labels = wine.target

x_train, x_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.3,
    random_state=15
)

tree_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4
)

tree_model.fit(x_train, y_train)

tree_predictions = tree_model.predict(x_test)

print(
    "Decision Tree Accuracy:",
    accuracy_score(y_test, tree_predictions)
)


print("\n================================")
print("RANDOM FOREST CLASSIFIER")
print("================================")

forest_model = RandomForestClassifier(
    n_estimators=50,
    random_state=15
)

forest_model.fit(x_train, y_train)

forest_predictions = forest_model.predict(x_test)

print(
    "Random Forest Accuracy:",
    accuracy_score(y_test, forest_predictions)
)

plt.figure(figsize=(8, 5))

plt.bar(
    range(len(forest_model.feature_importances_)),
    forest_model.feature_importances_
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Feature Number")
plt.ylabel("Importance")
plt.show()


print("\n================================")
print("SUPPORT VECTOR MACHINE")
print("================================")

svm_model = SVC(
    kernel="linear"
)

svm_model.fit(x_train, y_train)

svm_predictions = svm_model.predict(x_test)

print(
    "SVM Accuracy:",
    accuracy_score(y_test, svm_predictions)
)


print("\n================================")
print("PRINCIPAL COMPONENT ANALYSIS")
print("================================")

pca_model = PCA(n_components=2)

compressed_data = pca_model.fit_transform(features)

plt.figure(figsize=(8, 5))

plt.scatter(
    compressed_data[:, 0],
    compressed_data[:, 1],
    c=labels,
    cmap="plasma"
)

plt.title("PCA Visualization")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()


print("\n================================")
print("Q-LEARNING EXAMPLE")
print("================================")

size = 4

q_values = np.zeros((size, size, 4))

learning_rate = 0.2
discount = 0.8
exploration = 0.1

target = (3, 3)

for episode in range(300):

    current = [0, 0]

    while tuple(current) != target:

        if np.random.random() < exploration:
            move = np.random.randint(4)
        else:
            move = np.argmax(
                q_values[current[0], current[1]]
            )

        nxt = current.copy()

        if move == 0 and current[0] > 0:
            nxt[0] -= 1

        elif move == 1 and current[0] < 3:
            nxt[0] += 1

        elif move == 2 and current[1] > 0:
            nxt[1] -= 1

        elif move == 3 and current[1] < 3:
            nxt[1] += 1

        reward = 20 if tuple(nxt) == target else -1

        best_future = np.max(
            q_values[nxt[0], nxt[1]]
        )

        q_values[current[0], current[1], move] += (
            learning_rate *
            (
                reward +
                discount * best_future -
                q_values[current[0], current[1], move]
            )
        )

        current = nxt

print("Q-Learning Training Finished")


print("\n================================")
print("LSTM CELL DEMONSTRATION")
print("================================")

input_units = 4
hidden_units = 5

sample_input = np.random.randn(input_units)

previous_hidden = np.random.randn(hidden_units)

weights = np.random.randn(
    hidden_units,
    input_units + hidden_units
)

combined_vector = np.concatenate(
    (previous_hidden, sample_input)
)

forget_output = 1 / (
    1 + np.exp(
        -np.dot(weights, combined_vector)
    )
)

print("Forget Gate Values")
print(forget_output)


print("\n================================")
print("Q NETWORK DEMONSTRATION")
print("================================")

input_layer = 5
hidden_layer = 10
output_layer = 3

layer1 = np.random.randn(
    input_layer,
    hidden_layer
)

layer2 = np.random.randn(
    hidden_layer,
    output_layer
)

state = np.random.randn(
    1,
    input_layer
)

hidden_output = np.maximum(
    0,
    np.dot(state, layer1)
)

network_output = np.dot(
    hidden_output,
    layer2
)

print("Predicted Q Values")
print(network_output)
