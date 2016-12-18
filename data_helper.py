#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-12-6 下午2:54
import pickle
import numpy as np
from tensorflow.contrib import learn as tf_learn


class QaSample(object):
    def __init__(self, q_id, question, a_id, answer, label=None, score=0):
        self.q_id = q_id
        self.question = question
        self.a_id = a_id
        self.answer = answer
        self.label = int(label)
        self.score = float(score)


def load_qa_data(fname):
    with open(fname, 'r') as fin:
        for line in fin:
            try:
                q_id, question, a_id, answer, label = line.strip().split('\t')
            except ValueError:
                q_id, question, a_id, answer = line.strip().split('\t')
                label = 0
            yield QaSample(q_id, question, a_id, answer, label)


def get_final_rank(scored_samples):
    sample_final_rank = []
    same_q_samples = []
    for sample in scored_samples:
        if len(same_q_samples) == 0 or sample.q_id == same_q_samples[0].q_id:
            same_q_samples.append(sample)
        else:
            sorted_samples = sorted(same_q_samples, key=lambda s: s.score, reverse=True)
            sample_rank_map = {sample: rank for rank, sample in enumerate(sorted_samples)}
            for same_q_sample in same_q_samples:
                sample_final_rank.append((same_q_sample, sample_rank_map[same_q_sample]))
            same_q_samples = [sample]
    if len(same_q_samples) > 0:
        sorted_samples = sorted(same_q_samples, key=lambda s: s.score, reverse=True)
        sample_rank_map = {sample: rank for rank, sample in enumerate(sorted_samples)}
        for same_q_sample in same_q_samples:
            sample_final_rank.append((same_q_sample, sample_rank_map[same_q_sample]))
    return sample_final_rank


class DataHelper(object):
    def __init__(self):
        self.max_q_length = None
        self.max_a_length = None
        self.vocab = None
        self.embeddings = None
        self.vocab_processor = None
        self.train_triplets = None
        self.dev_samples = None
        self.dev_data = None
        self.test_samples = None
        self.test_data = None

    def build(self, embedding_file, train_file, dev_file, test_file=None):
        # loading all data
        train_samples = list(load_qa_data(train_file))
        dev_samples = list(load_qa_data(dev_file))
        if test_file:
            test_samples = list(load_qa_data(test_file))
        else:
            test_samples = []
        self.max_q_length = max([len(sample.question.split())
                                 for sample in train_samples + dev_samples + test_samples if sample.label == 1])
        self.max_a_length = max([len(sample.answer.split())
                                 for sample in train_samples + dev_samples + test_samples if sample.label == 1])
        print('Max question length: {}, max answer length: {}'.format(self.max_q_length, self.max_a_length))
        self.vocab, self.embeddings = self.load_embeddings(embedding_file, train_samples + dev_samples + test_samples)
        self.build_vocab_processor(texts=self.vocab, max_length=max(self.max_q_length, self.max_a_length))
        self.embeddings = np.concatenate(([[0] * self.embeddings.shape[1]], self.embeddings))

    def save(self, filename):
        print('Save data_helper to {}'.format(filename))
        info = {
            'max_q_length': self.max_q_length,
            'max_a_length': self.max_a_length,
            'vocab': self.vocab,
            'embeddings': self.embeddings,
            'vocab_processor': self.vocab_processor
        }
        with open(filename, 'wb') as fout:
            pickle.dump(info, fout)

    def restore(self, filename):
        print('Restore data_helper from {}'.format(filename))
        with open(filename, 'rb') as fin:
            info = pickle.load(fin)
        self.max_q_length = info['max_q_length']
        self.max_a_length = info['max_a_length']
        self.vocab = info['vocab']
        self.embeddings = info['embeddings']
        self.vocab_processor = info['vocab_processor']

    def load_embeddings(self, embedding_file, samples):
        print('Load embeddings from {}'.format(embedding_file))
        corpus_words = set([word for sample in samples for word in sample.question.split() + sample.answer.split()])
        vocab = []
        embeddings = []
        with open(embedding_file, 'r') as fin:
            for line in fin:
                try:
                    line_info = line.strip().split()
                    word = line_info[0]
                    embedding = [float(val) for val in line_info[1:]]
                    if word in corpus_words:
                        vocab.append(word)
                        embeddings.append(embedding)
                except:
                    # print('Error while loading line: {}'.format(line.strip()))
                    pass
        print('Vocabulary size: {}'.format(len(vocab)))
        return vocab, np.array(embeddings)

    def build_vocab_processor(self, texts, max_length):
        print('Build vocab_processor')
        self.vocab_processor = tf_learn.preprocessing.VocabularyProcessor(max_document_length=max_length)
        self.vocab_processor.fit(texts)

    def gen_train_batches(self, batch_size):
        data_size = len(self.train_triplets)
        num_batches = int((data_size - 1) / batch_size) + 1
        for batch_num in range(num_batches):
            start_index = batch_size * batch_num
            end_index = min(batch_size * (batch_num + 1), data_size)
            batch = self.train_triplets[start_index: end_index]
            yield batch

    def prepare_train_triplets(self, triplets_file):
        self.train_triplets = []
        with open(triplets_file, 'r') as fin:
            for line in fin:
                question, pos_ans, neg_ans = line.strip().split('\t')
                question_ids = list(self.vocab_processor.transform([question]))[0][:self.max_q_length]
                pos_ans_ids = list(self.vocab_processor.transform([pos_ans]))[0][:self.max_a_length]
                neg_ans_ids = list(self.vocab_processor.transform([neg_ans]))[0][:self.max_a_length]
                self.train_triplets.append((question_ids, pos_ans_ids, neg_ans_ids))

    def prepare_dev_data(self, dev_file):
        self.dev_samples = list(load_qa_data(dev_file))
        self.dev_data = []
        for sample in self.dev_samples:
            question_ids = list(self.vocab_processor.transform([sample.question]))[0][:self.max_q_length]
            answer_ids = list(self.vocab_processor.transform([sample.answer]))[0][:self.max_a_length]
            self.dev_data.append((question_ids, answer_ids))

    def prepare_test_data(self, test_file):
        self.test_samples = list(load_qa_data(test_file))
        self.test_data = []
        for sample in self.test_samples:
            question_ids = list(self.vocab_processor.transform([sample.question]))[0][:self.max_q_length]
            answer_ids = list(self.vocab_processor.transform([sample.answer]))[0][:self.max_a_length]
            self.test_data.append((question_ids, answer_ids))
