import re
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from colorama import Fore


def remove_html_tags(text):
    tag_regexp = re.compile(r'<[^>]+>')
    return tag_regexp.sub(" ", text)


def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]

    return " ".join(filtered_words)


def preprocess_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    text = text.replace("\n", " ")
    text = re.sub(r"'s", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)  # remove all chars except letters
    text = remove_stop_words(text)

    return text


def pad_vect(vect, max_length):
    for i, datum in enumerate(vect):
        if len(datum) > max_length:
            vect[i] = datum[:max_length]
        else:
            vect[i] = np.concatenate([datum, np.zeros(max_length - len(datum))])

    return vect


def create_embedding_matrix(word_index):
    file_name = "glove.6B.100d.txt"
    dim = 100
    vocab_size = len(word_index) + 1

    embedding_matrix = np.zeros((vocab_size, dim))

    with open(file_name) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                id = word_index[word]
                embedding_matrix[id] = np.array(vector)

    return embedding_matrix


if __name__ == '__main__':
    dataset = pd.read_csv('data.csv')
    print("Dataset preview:")
    print(dataset.head())
    print(f"Shape: {dataset.shape}")
    print(f"Missing values: {dataset.isnull().values.any()}")

    len_review_list = [len(review) for review in dataset['review']]
    max_review_len = max(len_review_list)

    print(f"Max. review length: {max_review_len}")

    # preprocessing
    X = [preprocess_text(review) for review in list(dataset['review'])]
    y = [1 if sentiment == "positive" else 0 for sentiment in list(dataset['sentiment'])]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # text to sequence
    word_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    word_tokenizer.fit_on_texts(X_train)

    num_unique_words = len(word_tokenizer.word_index) + 1
    print(f"Unique words: {num_unique_words}")

    X_train = word_tokenizer.texts_to_sequences(X_train)
    X_test = word_tokenizer.texts_to_sequences(X_test)

    # padding
    input_layer_len = max_review_len

    X_train = pad_vect(X_train, input_layer_len)
    X_test = pad_vect(X_test, input_layer_len)

    # embedding
    embedding_matrix = create_embedding_matrix(word_tokenizer.word_index)

    # cnn model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=num_unique_words, output_dim=100, weights=[embedding_matrix],
                                  trainable=False),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # training
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
    model = tf.keras.models.load_model("my_model.keras")

    # testing
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"Test accuracy: {round(accuracy, 4) * 100} %")

    # save model
    model.save('my_model.keras')

    # unseen data
    reviews = [
        "Amazing! The plot was captivating and the acting superb. A must-watch for everyone.",
        "Terrible movie. The storyline was confusing and the acting was subpar. Waste of time.",
        "Loved every minute of it! Great performances and a thrilling script.",
        "Awful! The movie was boring and poorly directed. I could not finish it.",
        "Fantastic! Engaging from start to finish. Highly recommend it to all."
    ]

    X_unseen = [preprocess_text(review) for review in reviews]
    X_unseen = word_tokenizer.texts_to_sequences(X_unseen)
    X_unseen = pad_vect(X_unseen, input_layer_len)
    X_unseen = np.array(X_unseen)

    Y_unseen = model.predict(X_unseen)


    def get_sentiment(y):
        if y >= 0.5:
            return Fore.GREEN + "Positive"
        else:
            return Fore.RED + "Negative"

    print(f"\n\nUnseen reviews:")
    for review, y in zip(reviews, Y_unseen):
        print(f"{Fore.WHITE + review} {get_sentiment(y)}")

