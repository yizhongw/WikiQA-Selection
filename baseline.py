#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-12-15 下午4:32
import string
from data_helper import load_qa_data, get_final_rank
from nltk.corpus import stopwords


def word_matching_rank(test_file, output_file):
    puncts = set(list(string.punctuation))
    stop_words = set(stopwords.words())
    samples = list(load_qa_data(test_file))
    for sample in samples:
        q_words = [word for word in sample.question.split() if word not in puncts and word not in stop_words]
        a_words = [word for word in sample.answer.split() if word not in puncts and word not in stop_words]
        cooccur_cnt = sum([word in q_words for word in a_words])
        sample.score = cooccur_cnt
    with open(output_file, 'w') as fout:
        for sample, rank in get_final_rank(samples):
            fout.write('{}\t{}\t{}\n'.format(sample.q_id, sample.a_id, rank))


def do_nothing(test_file, output_file):
    samples = list(load_qa_data(test_file))
    with open(output_file, 'w') as fout:
        for sample, rank in get_final_rank(samples):
            fout.write('{}\t{}\t{}\n'.format(sample.q_id, sample.a_id, rank))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_matching', action='store_true')
    parser.add_argument('--nothing', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.word_matching:
        word_matching_rank('data/lemmatized/WikiQA-dev.tsv', 'data/output/WikiQA-dev.rank')
    if args.nothing:
        do_nothing('data/lemmatized/WikiQA-dev.tsv', 'data/output/WikiQA-dev.rank')
