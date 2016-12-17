#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-12-6 下午2:54
import os
import argparse
import numpy as np
import tensorflow as tf
from cnn import QaCNN
from data_helper import DataHelper
from data_helper import get_final_rank

embedding_file = 'data/embeddings/glove.6B.300d.txt'
train_file = 'data/lemmatized/WikiQA-train.tsv'
dev_file = 'data/lemmatized/WikiQA-dev.tsv'
test_file = 'data/lemmatized/WikiQA-test.tsv'
train_triplets_file = 'data/lemmatized/WikiQA-train-triplets.tsv'


def prepare_helper():
    data_helper = DataHelper()
    data_helper.build(embedding_file, train_file, dev_file, test_file)
    data_helper.save('data/model/data_helper_info.bin')


def train_cnn():
    data_helper = DataHelper()
    data_helper.restore('data/model/data_helper_info.bin')
    data_helper.prepare_train_triplets('data/lemmatized/WikiQA-train-triplets.tsv')
    data_helper.prepare_dev_data('data/lemmatized/WikiQA-dev.tsv')
    cnn_model = QaCNN(
        q_length=data_helper.max_q_length,
        a_length=data_helper.max_a_length,
        word_embeddings=data_helper.embeddings,
        filter_sizes=[2, 3, 4],
        num_filters=3,
        margin=1,
        l2_reg_lambda=0
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_op = optimizer.minimize(cnn_model.loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(50):
            for batch in data_helper.gen_train_batches(batch_size=2):
                q_batch, pos_a_batch, neg_a_batch = zip(*batch)
                sess.run(train_op, feed_dict={cnn_model.question: q_batch,
                                              cnn_model.pos_answer: pos_a_batch,
                                              cnn_model.neg_answer: neg_a_batch,
                                              cnn_model.dropout_keep_prob: 1.0
                                              })
            q_train, pos_a_train, neg_a_train = zip(*data_helper.train_triplets)
            loss = sess.run(cnn_model.loss, feed_dict={cnn_model.question: q_train,
                                                       cnn_model.pos_answer: pos_a_train,
                                                       cnn_model.neg_answer: neg_a_train,
                                                       cnn_model.dropout_keep_prob: 1.0
                                                       })
            print('Loss in epoch {}: {}'.format(epoch, loss))

            q_dev, q_ans = zip(*data_helper.dev_data)
            similarity_scores = sess.run(cnn_model.pos_similarity, feed_dict={cnn_model.question: q_dev,
                                                                       cnn_model.pos_answer: q_ans,
                                                                       cnn_model.neg_answer: q_ans,
                                                                       cnn_model.dropout_keep_prob: 1.0
                                                                       })
            for sample, similarity_score in zip(data_helper.dev_samples, similarity_scores):
                sample.score = similarity_score
            with open('data/output/WikiQA-dev-{}.rank'.format(epoch), 'w') as fout:
                for sample, rank in get_final_rank(data_helper.dev_samples):
                    fout.write('{}\t{}\t{}\n'.format(sample.q_id, sample.a_id, rank))
            os.system('python3 eval.py data/output/WikiQA-dev-{}.rank data/raw/WikiQA-dev.tsv'.format(epoch))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true', help='whether to prepare data helper')
    parser.add_argument('--train', action='store_true', help='train a model for answer selection')
    parser.add_argument('--test', action='store_true', help='generate the rank for test file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.prepare:
        prepare_helper()
    if args.train:
        train_cnn()




# with tf.Graph().as_default():
#     sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
#     with tf.Session(config=sess_config) as sess:
#
#         model = QaCNN(n_class=2, q_length=data_helper.max_q_length, a_length=data_helper.max_a_length)
#
#         global_step = tf.Variable(0, name='global_step', trainable=False)
#         optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
#         grads_and_vars = optimizer.compute_gradients(model.loss)
#         train_op = optimizer.apply_gradients(grads_and_vars, global_step)
#
#         loss_summary = tf.scalar_summary('loss', model.loss)
#         acc_summary = tf.scalar_summary('accuracy', model.accuracy)
#
#         train_summary_op = tf.merge_summary([loss_summary, acc_summary])
#         train_summary_writer = tf.train.SummaryWriter('data/run/summaries/train', sess.graph)
#
#         dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
#         dev_summary_writer = tf.train.SummaryWriter('data/run/summaries/dev', sess.graph)
#
#         checkpoint_dir = os.path.abspath('data/run/checkpoints/')
#         checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
#         saver = tf.train.Saver(tf.all_variables())
#
#         sess.run(tf.initialize_all_variables())
#
#         def train_step(q_batch, a_batch, y_batch):
#             feed_dict = {
#                 model.input_q: q_batch,
#                 model.input_a: a_batch,
#                 model.input_y: y_batch
#             }
#             _, step, summaries, loss, accuracy = sess.run(
#                 [train_op, global_step, train_summary_op, model.loss, model.accuracy],
#                 feed_dict=feed_dict
#             )
#             print('Step {}: loss {}, acc {}.'.format(step, loss, accuracy))
#             train_summary_writer.add_summary(summaries, step)
#
#         def dev_step(q_dev, a_dev, y_dev):
#             feed_dict = {
#                 model.input_q: q_dev,
#                 model.input_a: a_dev,
#                 model.input_y: y_dev
#             }
#             step, summaries, loss, accuracy = sess.run(
#                 [global_step, dev_summary_op, model.loss, model.accuracy],
#                 feed_dict=feed_dict
#             )
#             print('Step {}: loss {}, acc {}.'.format(step, loss, accuracy))
#             dev_summary_writer.add_summary(summaries, step)
#
#         train_batches = data_helper.gen_train_batches(batch_size=10, num_epochs=5)
#         q_dev, a_dev, y_dev = data_helper.get_dev_data()
#         for q_batch, a_batch, y_batch in train_batches:
#             train_step(q_batch, a_batch, y_batch)
#             cur_step = tf.train.global_step(sess, global_step)
#             if cur_step % 100 == 0:
#                 print('Evaluation:')
#                 dev_step(q_dev, a_dev, y_dev)
#             if cur_step % 1000 == 0:
#                 path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
#                 print('Done with model checkpoint saving.')
