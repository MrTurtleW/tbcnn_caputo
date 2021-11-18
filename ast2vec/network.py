import math

import tensorflow as tf

from .data_loader import node_map


def build_net(batch_size, num_feats, hidden_size):
    with tf.name_scope('network'):
        with tf.name_scope('inputs'):
            inputs = tf.placeholder(tf.int32, shape=[batch_size, ], name='inputs')
            labels = tf.placeholder(tf.int32, shape=[batch_size, ], name='labels')

            # embeddings to learn
            embeddings = tf.Variable(
                tf.random_uniform([len(node_map), num_feats]), name='embeddings'
            )

            embed = tf.nn.embedding_lookup(embeddings, inputs)
            onehot_labels = tf.one_hot(labels, len(node_map), dtype=tf.float32)

        with tf.name_scope('hidden'):
            weights = tf.Variable(
                tf.truncated_normal(
                    [num_feats, hidden_size], stddev=1.0 / math.sqrt(num_feats)
                ),
                name='weights'
            )

            biases = tf.Variable(
                tf.zeros((hidden_size,)),
                name='biases'
            )

            hidden = tf.tanh(tf.matmul(embed, weights) + biases)

        with tf.name_scope('softmax'):
            weights = tf.Variable(
                tf.truncated_normal(
                    [hidden_size, len(node_map)],
                    stddev=1.0 / math.sqrt(hidden_size)
                ),
                name='weights'
            )
            biases = tf.Variable(
                tf.zeros((len(node_map),), name='biases')
            )

            logits = tf.nn.xw_plus_b(hidden, weights, biases)

        with tf.name_scope('error'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=onehot_labels, logits=logits, name='cross_entropy'
            )

            loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        # batch_X, batch_y, output, loss
        return inputs, labels, embeddings, loss
