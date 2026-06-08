import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


print("\n================================")
print("IMAGE FILTERING")
print("================================")

sample_image = np.random.rand(50, 50)

smooth_filter = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) / 9

filtered_image = np.zeros_like(sample_image)

for row in range(1, 49):
    for col in range(1, 49):

        patch = sample_image[
            row - 1:row + 2,
            col - 1:col + 2
        ]

        filtered_image[row, col] = np.sum(
            patch * smooth_filter
        )

plt.imshow(filtered_image, cmap="gray")
plt.title("Smoothed Image")
plt.axis("off")
plt.show()


print("\n================================")
print("COMPUTER VISION FEATURE EXTRACTION")
print("================================")

vertical_kernel = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

feature_image = np.zeros_like(sample_image)

for row in range(1, 49):
    for col in range(1, 49):

        patch = sample_image[
            row - 1:row + 2,
            col - 1:col + 2
        ]

        feature_image[row, col] = np.sum(
            patch * vertical_kernel
        )

plt.imshow(feature_image, cmap="gray")
plt.title("Extracted Features")
plt.axis("off")
plt.show()


print("\n================================")
print("CNN OPERATION")
print("================================")

cnn_input = np.random.rand(32, 32)

cnn_filter = np.random.rand(5, 5)

cnn_output = np.zeros((28, 28))

for row in range(28):
    for col in range(28):

        region = cnn_input[
            row:row + 5,
            col:col + 5
        ]

        cnn_output[row, col] = np.sum(
            region * cnn_filter
        )

activation_map = np.maximum(
    cnn_output,
    0
)

print("Activation Map Shape:")
print(activation_map.shape)

plt.imshow(activation_map, cmap="gray")
plt.title("CNN Activation Map")
plt.axis("off")
plt.show()


print("\n================================")
print("IMAGE CLASSIFICATION")
print("================================")

image_features = np.random.rand(
    400,
    128
)

image_labels = np.random.randint(
    0,
    3,
    400
)

x_train, x_test, y_train, y_test = train_test_split(
    image_features,
    image_labels,
    test_size=0.25,
    random_state=25
)

classifier = DecisionTreeClassifier()

classifier.fit(
    x_train,
    y_train
)

predictions = classifier.predict(
    x_test
)

print(
    "Classification Accuracy:",
    accuracy_score(
        y_test,
        predictions
    )
)


print("\n================================")
print("RECURRENT NEURAL NETWORK")
print("================================")

input_neurons = 4
hidden_neurons = 6

input_weights = np.random.randn(
    hidden_neurons,
    input_neurons
)

recurrent_weights = np.random.randn(
    hidden_neurons,
    hidden_neurons
)

hidden_state = np.zeros(
    (hidden_neurons, 1)
)

time_steps = [
    np.random.randn(
        input_neurons,
        1
    )
    for _ in range(6)
]

for step in time_steps:

    hidden_state = np.tanh(
        np.dot(
            input_weights,
            step
        ) +
        np.dot(
            recurrent_weights,
            hidden_state
        )
    )

print("Hidden State Output:")
print(hidden_state)