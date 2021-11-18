import os
import sys

import numpy as np

import tensorflow as tf

cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '..', '..'))

from config import tbcnn_caputo_ckpt_filename
from tbcnn.data_loader import DataLoader
from tbcnn.network import build_net
from tbcnn.caputo.train import num_features, label_size


def test(proba=True):
    nodes, children, labels, output, loss = build_net(num_features, label_size)

    sess = tf.Session()
    saver = tf.train.Saver()

    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(tbcnn_caputo_ckpt_filename)))

    data_loader = DataLoader()

    pred = []
    y_test = []
    for test_nodes, test_children, test_labels in data_loader.batch_data('test', batch_size=1):
        _out = sess.run(output, feed_dict={
            nodes: test_nodes, children: test_children, labels: test_labels})

        if proba:
            pred.extend(np.max(_out, axis=1))
        else:
            pred.extend(np.argmax(_out, axis=1))
        y_test.extend(np.argmax(test_labels, axis=1))

    return 'tbcnn-caputo', y_test, pred


if __name__ == '__main__':
    test()