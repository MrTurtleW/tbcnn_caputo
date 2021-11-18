import os
import sys

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '..'))

from data.parse_ast import Node

from compare.svm import main as svm
from compare.bayes import main as bayes
from compare.random_forest import main as random_forest
from tbcnn.integer.test import test as tbcnn
from tbcnn.caputo.test import test as tbcnn_caputo


def main():
    plt.figure()
    for model in (tbcnn, tbcnn_caputo, svm, bayes, random_forest):
        name, y_test, pred = model()
        fpr, tpr, _ = roc_curve(y_test, pred)
        plt.plot(fpr, tpr, label='%s (area = %.4f)' % (name, auc(fpr, tpr)))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    plt.show()


if __name__ == '__main__':
    main()