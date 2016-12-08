#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-12-6 下午2:54
import os
import numpy as np
import tensorflow as tf
from cnn import QaCNN
from data_helper import DataHelper

train_file = 'data/lemmatized/WikiQA-train.tsv'
dev_file = 'data/lemmatized/WikiQA-dev.tsv'
test_file = 'data/lemmatized/WikiQA-test.tsv'
data_helper = DataHelper(train_file, dev_file, test_file)

q_dev, a_dev, y_dev = data_helper.get_dev_data()

with tf.Graph().as_default():
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.Session(config=sess_config) as sess:

        model = QaCNN(n_class=2, q_length=data_helper.max_q_length, a_length=data_helper.max_a_length)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

        loss_summary = tf.scalar_summary('loss', model.loss)
        acc_summary = tf.scalar_summary('accuracy', model.accuracy)

        train_summary_op = tf.merge_summary([loss_summary, acc_summary])
        train_summary_writer = tf.train.SummaryWriter('data/run/summaries/train', sess.graph)

        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_writer = tf.train.SummaryWriter('data/run/summaries/dev', sess.graph)

        checkpoint_dir = os.path.abspath('data/run/checkpoints/')
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        saver = tf.train.Saver(tf.all_variables())

        sess.run(tf.initialize_all_variables())

        def train_step(q_batch, a_batch, y_batch):
            feed_dict = {
                model.input_q: q_batch,
                model.input_a: a_batch,
                model.input_y: y_batch
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict=feed_dict
            )
            print('Step {}: loss {}, acc {}.'.format(step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(q_dev, a_dev, y_dev):
            feed_dict = {
                model.input_q: q_dev,
                model.input_a: a_dev,
                model.input_y: y_dev
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy],
                feed_dict=feed_dict
            )
            print('Step {}: loss {}, acc {}.'.format(step, loss, accuracy))
            dev_summary_writer.add_summary(summaries, step)

        train_batches = data_helper.gen_train_batches(batch_size=10, num_epochs=5)
        q_dev, a_dev, y_dev = data_helper.get_dev_data()
        for q_batch, a_batch, y_batch in train_batches:
            train_step(q_batch, a_batch, y_batch)
            cur_step = tf.train.global_step(sess, global_step)
            if cur_step % 100 == 0:
                print('Evaluation:')
                dev_step(q_dev, a_dev, y_dev)
            if cur_step % 1000 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
                print('Done with model checkpoint saving.')
