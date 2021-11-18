import os
import sys
import time
import math
import shutil

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '..', '..'))
from ast2vec.train import num_features
from data.parse_ast import Node
from config import tbcnn_caputo_log_dir, tbcnn_caputo_ckpt_filename
from tbcnn.data_loader import DataLoader
from tbcnn.network import build_net

label_size = 2

epochs = 5
alpha = 0.9
delta = 1e-7
batch_size = 10
learning_rate = 0.1
checkpoint_step = 1000


def train():
    data_loader = DataLoader()
    nodes, children, labels, output, loss = build_net(num_features, label_size)

    gamma = math.gamma(2 - alpha)
    with tf.name_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)

        assigns = []
        for g, w in grads_and_vars:
            name, _ = w.name.split(':')
            g_k_1 = tf.get_variable(name + 'g_k_1', shape=g.shape, dtype=tf.float32,
                                    initializer=xavier_initializer(), trainable=False)
            w_k = tf.get_variable(name + 'w_k', shape=w.shape, dtype=tf.float32,
                                  initializer=xavier_initializer(), trainable=False)
            fo_w = learning_rate * g_k_1 * tf.pow(tf.abs(w - w_k) + delta, 1 - alpha) / gamma
            assigns.append(tf.assign(w, w - learning_rate * fo_w))
            assigns.append(tf.assign(g_k_1, g))
            assigns.append(tf.assign(w_k, w))

    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('saver'):
        saver = tf.train.Saver()

        if os.path.exists(tbcnn_caputo_ckpt_filename):
            shutil.rmtree(tbcnn_caputo_ckpt_filename)
        os.makedirs(tbcnn_caputo_ckpt_filename)

    writer = tf.summary.FileWriter(tbcnn_caputo_log_dir)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar('accuracy', accuracy)
    write_op = tf.summary.merge_all()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    total_batch = 0
    for epoch in range(epochs):
        time_start = time.time()
        for batch, data in enumerate(data_loader.batch_data('train', batch_size=batch_size)):
            batch_nodes, batch_children, batch_labels = data
            _loss, _acc, summary = sess.run([loss, accuracy, write_op], feed_dict={
                nodes: batch_nodes, children: batch_children, labels: batch_labels})
            writer.add_summary(summary, total_batch)
            sess.run(assigns, feed_dict={
                nodes: batch_nodes, children: batch_children, labels: batch_labels})

            total_batch += 1
            if batch % 100 == 0:
                time_end = time.time()
                print('Epoch: {}, train batch: {}, loss: {}, acc: {}, time: {}'.format(epoch, batch, _loss, _acc,
                                                                                       time_end - time_start))
                time_start = time_end

            if total_batch % checkpoint_step == 0:
                saver.save(sess, os.path.join(tbcnn_caputo_ckpt_filename), total_batch)
                print('checkpoint saved.')

        test_loss = []
        test_accuracy = []
        for test_nodes, test_children, test_labels in data_loader.batch_data('test', batch_size=1):
            _loss, _acc = sess.run([loss, accuracy], feed_dict={
                nodes: test_nodes, children: test_children, labels: test_labels})

            test_loss.append(_loss)
            test_accuracy.append(_acc)

        avg_loss = sum(test_loss) / len(test_loss)
        avg_accuracy = sum(test_accuracy) / len(test_accuracy)
        print('Test loss: {:.2f}, accuracy: {:.2f}'.format(avg_loss, avg_accuracy))


if __name__ == '__main__':
    train()
