#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-12-6 下午2:54
import argparse
import tensorflow as tf
from cnn import QaCNN
from data_helper import DataHelper
from data_helper import get_final_rank
from eval import eval_map_mrr

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
    data_helper.prepare_test_data('data/lemmatized/WikiQA-test.tsv')
    cnn_model = QaCNN(
        q_length=data_helper.max_q_length,
        a_length=data_helper.max_a_length,
        word_embeddings=data_helper.embeddings,
        filter_sizes=[1, 2, 3, 5, 7, 9],
        num_filters=128,
        margin=0.25,
        l2_reg_lambda=0
    )

    global_step = tf.Variable(0, name='global_step', trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_op = optimizer.minimize(cnn_model.loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(50):
            train_loss = 0
            for batch in data_helper.gen_train_batches(batch_size=15):
                q_batch, pos_a_batch, neg_a_batch = zip(*batch)
                _, loss = sess.run([train_op, cnn_model.loss], feed_dict={cnn_model.question: q_batch,
                                                                          cnn_model.pos_answer: pos_a_batch,
                                                                          cnn_model.neg_answer: neg_a_batch,
                                                                          })
                train_loss += loss

                cur_step = tf.train.global_step(sess, global_step)
                if cur_step % 10 == 0:
                    # print('Loss: {}'.format(train_loss))
                    # test on dev set
                    q_dev, ans_dev = zip(*data_helper.dev_data)
                    similarity_scores = sess.run(cnn_model.pos_similarity, feed_dict={cnn_model.question: q_dev,
                                                                                      cnn_model.pos_answer: ans_dev,
                                                                                      cnn_model.neg_answer: ans_dev,
                                                                                      })
                    for sample, similarity_score in zip(data_helper.dev_samples, similarity_scores):
                        sample.score = similarity_score
                    with open('data/output/WikiQA-dev.rank'.format(epoch), 'w') as fout:
                        for sample, rank in get_final_rank(data_helper.dev_samples):
                            fout.write('{}\t{}\t{}\n'.format(sample.q_id, sample.a_id, rank))
                    dev_MAP, dev_MRR = eval_map_mrr('data/output/WikiQA-dev.rank'.format(epoch), 'data/raw/WikiQA-dev.tsv')
                    # print('Dev MAP: {}, MRR: {}'.format(dev_MAP, dev_MRR))

                    # test on test set
                    q_test, ans_test = zip(*data_helper.test_data)
                    similarity_scores = sess.run(cnn_model.pos_similarity, feed_dict={cnn_model.question: q_test,
                                                                                      cnn_model.pos_answer: ans_test,
                                                                                      cnn_model.neg_answer: ans_test,
                                                                                      })
                    for sample, similarity_score in zip(data_helper.test_samples, similarity_scores):
                        # print('{}\t{}\t{}'.format(sample.q_id, sample.a_id, similarity_score))
                        sample.score = similarity_score
                    with open('data/output/WikiQA-test.rank', 'w') as fout:
                        for sample, rank in get_final_rank(data_helper.test_samples):
                            fout.write('{}\t{}\t{}\n'.format(sample.q_id, sample.a_id, rank))
                    test_MAP, test_MRR = eval_map_mrr('data/output/WikiQA-test.rank'.format(epoch), 'data/raw/WikiQA-test-gold.tsv')
                    # print('Test MAP: {}, MRR: {}'.format(test_MAP, test_MRR))
                    print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(epoch, cur_step, train_loss, dev_MAP, dev_MRR, test_MAP, test_MRR))
                    train_loss = 0


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
