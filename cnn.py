#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-12-6 下午8:32
import tensorflow as tf


class QaCNN(object):
    def __init__(self, q_length, a_length, word_embeddings, filter_sizes, num_filters, margin, l2_reg_lambda):

        self.question = tf.placeholder(tf.int32, [None, q_length], name='question')
        self.pos_answer = tf.placeholder(tf.int32, [None, a_length], name='pos_answer')
        self.neg_answer = tf.placeholder(tf.int32, [None, a_length], name='neg_answer')

        l2_reg_loss = tf.constant(0.0)

        vocab_size, embedding_size = word_embeddings.shape
        num_filters_total = num_filters * len(filter_sizes)

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embeddings = tf.get_variable("embeddings",
                                              shape=word_embeddings.shape,
                                              initializer=tf.constant_initializer(word_embeddings),
                                              trainable=True)
            self.embedded_q = tf.nn.embedding_lookup(self.embeddings, self.question)
            self.embedded_pos_a = tf.nn.embedding_lookup(self.embeddings, self.pos_answer)
            self.embedded_neg_a = tf.nn.embedding_lookup(self.embeddings, self.neg_answer)
            self.embedded_q_expanded = tf.expand_dims(self.embedded_q, -1)
            self.embedded_pos_a_expanded = tf.expand_dims(self.embedded_pos_a, -1)
            self.embedded_neg_a_expanded = tf.expand_dims(self.embedded_neg_a, -1)

        # conv-pool-drop for question
        pooled_q_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope('ques-conv-pool-{}'.format(filter_size)):
                # convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(self.embedded_q_expanded, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(h, ksize=[1, q_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')
                pooled_q_outputs.append(pooled)

        self.q_h_pool = tf.reshape(tf.concat(3, pooled_q_outputs), [-1, num_filters_total])

        # conv-pool-drop for positive answer
        pos_pooled_outputs = []
        neg_pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope('answ-conv-pool-{}'.format(filter_size)):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                # convolution layer
                pos_conv = tf.nn.conv2d(self.embedded_pos_a_expanded, W, strides=[1, 1, 1, 1], padding='VALID', name='pos-conv')
                pos_h = tf.nn.relu(tf.nn.bias_add(pos_conv, b), name='pos-relu')
                # max pooling layer
                pos_pooled = tf.nn.max_pool(pos_h, ksize=[1, a_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1],
                                            padding='VALID', name='pos-pool')
                pos_pooled_outputs.append(pos_pooled)

                neg_conv = tf.nn.conv2d(self.embedded_neg_a_expanded, W, strides=[1, 1, 1, 1], padding='VALID', name='neg-conv')
                neg_h = tf.nn.relu(tf.nn.bias_add(neg_conv, b), name='neg-relu')
                neg_pooled = tf.nn.max_pool(neg_h, ksize=[1, a_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1],
                                            padding='VALID', name='neg-pool')
                neg_pooled_outputs.append(neg_pooled)

        self.pos_h_pool = tf.reshape(tf.concat(3, pos_pooled_outputs), [-1, num_filters_total])

        self.neg_h_pool = tf.reshape(tf.concat(3, neg_pooled_outputs), [-1, num_filters_total])

        with tf.name_scope('similarity'):
            normalized_q_h_pool = tf.nn.l2_normalize(self.q_h_pool, dim=1)
            normalized_pos_h_pool = tf.nn.l2_normalize(self.pos_h_pool, dim=1)
            normalized_neg_h_pool = tf.nn.l2_normalize(self.neg_h_pool, dim=1)
            self.pos_similarity = tf.reduce_sum(tf.mul(normalized_q_h_pool, normalized_pos_h_pool), 1)
            self.neg_similarity = tf.reduce_sum(tf.mul(normalized_q_h_pool, normalized_neg_h_pool), 1)

        with tf.name_scope('loss'):
            original_loss = tf.reduce_sum(margin - self.pos_similarity + self.neg_similarity)
            self.loss = tf.cond(tf.less(0.0, original_loss), lambda: original_loss, lambda: tf.constant(0.0))

