#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-12-6 下午2:54
import random
import os
from tensorflow.contrib import learn as tf_learn


class Sample(object):
    def __init__(self, q_id, question, a_id, answer, label):
        self.q_id = q_id
        self.question = question
        self.a_id = a_id
        self.answer = answer
        self.label = label


class DataHelper(object):
    def __init__(self, train_file, dev_file, test_file=None):
        print('Loading data to DataHelper...')
        self.train_data = list(self.load_qa_data(train_file))
        self.dev_data = list(self.load_qa_data(dev_file))
        if test_file:
            self.test_data = list(self.load_qa_data(test_file))
        else:
            self.test_data = []

        self.qid_to_sample_map = {}
        for sample in self.train_data:
            if sample.q_id in self.qid_to_sample_map:
                self.qid_to_sample_map.append(sample)
            else:
                self.qid_to_sample_map = [sample]

        self.max_q_length = max([len(sample.question) for sample in self.train_data + self.dev_data + self.test_data])
        self.max_a_length = max([len(sample.answer) for sample in self.train_data + self.dev_data + self.test_data])

        if os.path.exists('data/run/vocab'):
            print('Loading DataHelper vocabulary...')
            self.vocab_processor = tf_learn.preprocessing.VocabularyProcessor.restore('data/run/vocab')
        else:
            print('Building DataHelper vocabulary...')
            self.vocab_processor = self.build_vocab()
            # self.vocab_processor.save('data/run/vocab')
        print('Vocabulary size: {}.'.format(len(self.vocab_processor.vocabulary_)))

    def build_vocab(self):
        vocab_processor = tf_learn.preprocessing.VocabularyProcessor(
            max_document_length=max(self.max_q_length, self.max_a_length))
        all_texts = (sample.question + sample.answer for sample in self.train_data + self.dev_data + self.test_data)
        vocab_processor.fit(all_texts)
        return vocab_processor

    def gen_train_batches(self, batch_size, num_epochs):
        for epoch in range(num_epochs):
            balanced_train_data = list(self.get_balanced_train_data())
            data_size = len(balanced_train_data)
            num_batches_per_epoch = int(data_size / batch_size) + 1
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_size * batch_num
                end_index = min(batch_size * batch_num + 1, data_size)
                sample_batch = balanced_train_data[start_index: end_index]
                yield self.split_samples(sample_batch)

    def get_dev_data(self):
        return self.split_samples(self.dev_data)

    def split_samples(self, samples):
        q_list, a_list, y_list = [], [], []
        for sample in samples:
            q_list.append(list(self.vocab_processor.transform(sample.question)))
            a_list.append(list(self.vocab_processor.transform(sample.answer)))
            y_list.append(sample.label)
        return q_list, a_list, y_list

    def get_balanced_train_data(self):
        for sample in self.train_data:
            if sample.label == 1:
                yield sample
                negative_samples = [it for it in self.qid_to_sample_map[sample.q_id] if it.label == 0]
                yield negative_samples[random.randint(0, len(negative_samples)-1)]

    @staticmethod
    def load_qa_data(fname):
        with open(fname, 'r') as fin:
            for line in fin:
                try:
                    q_id, question, a_id, answer, label = line.strip().split('\t')
                except ValueError:
                    q_id, question, a_id, answer = line.strip().split('\t')
                    label = 0
                yield Sample(q_id, question, a_id, answer, label)
