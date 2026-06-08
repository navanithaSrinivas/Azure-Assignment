import numpy as np

print("\n================================")
print("CHATBOT USING RNN AND LSTM CONCEPTS")
print("================================")

conversation_data = {
    "hello": "hi",
    "good morning": "good morning to you",
    "how are you": "i am doing well",
    "what is machine learning": "machine learning helps computers learn from data",
    "bye": "see you later"
}

words = set()

for question, answer in conversation_data.items():
    words.update(question.split())
    words.update(answer.split())

vocabulary = sorted(list(words))

word_index = {
    word: position
    for position, word in enumerate(vocabulary)
}

index_word = {
    position: word
    for word, position in word_index.items()
}

vocab_length = len(vocabulary)

input_weights = np.random.randn(
    12,
    vocab_length
)

hidden_weights = np.random.randn(
    12,
    12
)

output_weights = np.random.randn(
    vocab_length,
    12
)


def encode_word(word):

    vector = np.zeros(
        (vocab_length, 1)
    )

    if word in word_index:
        vector[
            word_index[word]
        ] = 1

    return vector


def process_sequence(sentence):

    hidden = np.zeros(
        (12, 1)
    )

    for token in sentence:

        encoded = encode_word(token)

        hidden = np.tanh(
            np.dot(
                input_weights,
                encoded
            ) +
            np.dot(
                hidden_weights,
                hidden
            )
        )

    prediction = np.dot(
        output_weights,
        hidden
    )

    return prediction


def generate_reply(message):

    message = message.lower()

    if message in conversation_data:
        return conversation_data[message]

    tokens = message.split()

    if len(tokens) == 0:
        return "please enter a message"

    process_sequence(tokens)

    return "i do not have a response for that"


print("\nVirtual Assistant Ready")
print("Type 'quit' to stop\n")

while True:

    user_message = input("User : ")

    if user_message.lower() == "quit":

        print("Bot  : Session Ended")
        break

    response = generate_reply(
        user_message
    )

    print("Bot  :", response)