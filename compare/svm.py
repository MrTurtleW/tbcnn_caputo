import os
import sys

from sklearn.svm import SVC


cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '..'))

from compare.tfidf import tfidf


def main(proba=True):
    X_train, X_test, y_train, y_test = tfidf()
    # C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0
    # kernel: 参数选择有rbf, linear, poly, Sigmoid, 默认的是"RBF"
    svm = SVC()
    svm.fit(X_train, y_train)

    if proba:
        pred = svm.decision_function(X_test)
    else:
        pred = svm.predict(X_test)

    return 'svm', y_test, pred


if __name__ == '__main__':
    main()
