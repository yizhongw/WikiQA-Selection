#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-12-6 下午8:32
import tensorflow as tf


class QaCNN(object):
    def __init__(self, n_class, q_length, a_length):
        pass
        self.loss = 0
        self.accuracy = 0

        self.input_q = tf.placeholder(tf.int32, [None, q_length], name='input_q')
        self.input_a = tf.placeholder(tf.int32, [None, a_length], name='input_a')
        self.input_y = tf.placeholder(tf.float32, [None, n_class], name='input_y')