import os
import sys
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '..'))

from config import black_output_path, white_output_path, tfidf_file

stop_word = [" ", ",", "+", "(", ")", ";", "\n", '"', "'", '{', '}', '[', ']', '=', '|', '.', '#', '<', '>', '^', '\t',
             '/', '?', '\\']


def tokenize(string):
    start_index = 0
    current_index = 0
    while current_index < len(string):
        if string[current_index] in stop_word:
            substr = string[start_index:current_index]
            if 3 < len(substr) < 20:
                yield substr
            start_index = current_index + 1

        current_index += 1


def load_keywords_from_dataset(path):
    result = []
    for filename in os.listdir(path):
        abs_filename = os.path.join(path, filename)
        print('processing {}'.format(abs_filename))
        try:
            with open(abs_filename, 'r', encoding='utf-8') as f:
                data = f.read()
        except UnicodeDecodeError:
            continue

        document = []
        for token in tokenize(data):
            document.append(token)

        result.append(document)

    return result


def load_dataset():
    if os.path.exists(tfidf_file):
        with open(tfidf_file, 'rb') as f:
            dataset, label = pickle.load(f)

        return dataset, label

    black = [" ".join(l) for l in load_keywords_from_dataset(black_output_path)]
    white = [" ".join(l) for l in load_keywords_from_dataset(white_output_path)]

    black_labels = [1] * len(black)
    white_labels = [0] * len(white)

    dataset = black + white
    labels = black_labels + white_labels

    with open(tfidf_file, 'wb') as f:
        pickle.dump((dataset, labels), f)

    return dataset, labels


def tfidf():
    dataset, labels = load_dataset()
    print("dataset size: {}".format(len(dataset)))
    print("label size: {}".format(len(labels)))

    vectorizer = TfidfVectorizer()
    tfidf_dataset = vectorizer.fit_transform(dataset)
    print("tfidf_dataset size: {}".format(tfidf_dataset.shape))
    X_train, X_test, y_train, y_test = train_test_split(tfidf_dataset, labels)
    print("X_train size: {}".format(X_train.shape))
    print("X_test size: {}".format(X_test.shape))
    print("y_train size: {}".format(len(y_train)))
    print("y_test size: {}".format(len(y_test)))

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    tfidf()
