import os
import sys

import numpy as np

from sklearn.naive_bayes import BernoulliNB

cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '..'))

from compare.tfidf import tfidf


def main(proba=True):
    X_train, X_test, y_train, y_test = tfidf()

    bernoulli = BernoulliNB()
    bernoulli.fit(X_train, y_train)

    if proba:
        pred = np.max(bernoulli.predict_proba(X_test), axis=1)
    else:
        pred = bernoulli.predict(X_test)
    return 'bayes', y_test, pred


if __name__ == '__main__':
    main()
