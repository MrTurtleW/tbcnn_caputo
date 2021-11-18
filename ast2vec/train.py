import os
import sys
import time
import pickle

import tensorflow as tf

cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '..'))

from config import embed_file, ast2vec_logdir
from data.parse_ast import Node
from ast2vec.network import build_net
from ast2vec.data_loader import batch_data

batch_size = 50
num_features = 30
hidden_size = 100

learning_rate = 0.001
epochs = 100


def train():
    inputs, labels, embeddings, loss = build_net(batch_size, num_features, hidden_size)

    with tf.name_scope('optimizer'):
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    embed_path = os.path.dirname(embed_file)
    if not os.path.exists(embed_path):
        os.makedirs(embed_path)

    tf.summary.scalar('loss', loss)
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(ast2vec_logdir)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    total_batch = 0
    for epoch in range(epochs):
        time_start = time.time()
        for batch, (batch_X, batch_Y) in enumerate(batch_data(batch_size=batch_size)):
            _, summary, _loss, embed = sess.run([train_op, summaries, loss, embeddings], feed_dict={
                inputs: batch_X, labels: batch_Y})

            writer.add_summary(summary, total_batch)

            total_batch += 1
            if batch % 100 == 0:
                time_end = time.time()
                print('Epoch: {}, train batch: {}, loss: {}, time: {}'.format(epoch, batch, _loss,
                                                                              time_end - time_start))
                time_start = time_end

        with open(embed_file, 'wb') as f:
            pickle.dump(embed, f)


if __name__ == '__main__':
    train()
