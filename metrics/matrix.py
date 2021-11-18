import os
import sys

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from prettytable import PrettyTable

cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '..'))

from data.parse_ast import Node

from compare.svm import main as svm
from compare.bayes import main as bayes
from compare.random_forest import main as random_forest
from tbcnn.integer.test import test as tbcnn
from tbcnn.caputo.test import test as tbcnn_caputo


def main():
    table = PrettyTable(['model', 'accuracy', 'precision', 'recall', 'f1-score'])
    for model in (tbcnn, tbcnn_caputo, svm, bayes, random_forest):
        name, y_test, pred = model(proba=False)
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        table.add_row([name, accuracy, precision, recall, f1])

    print(table)


if __name__ == '__main__':
    main()
