#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-12-6 下午3:00
import nltk

if __name__ == '__main__':
    wn_lemmatizer = nltk.stem.WordNetLemmatizer()
    data_sets = ['train', 'dev', 'test']
    for set_name in data_sets:
        fin_path = 'data/raw/WikiQA-{}.tsv'.format(set_name)
        fout_path = 'data/lemmatized/WikiQA-{}.tsv'.format(set_name)
        with open(fin_path, 'r') as fin, open(fout_path, 'w') as fout:
            fin.readline()
            for line in fin:
                line_info = line.strip().split('\t')
                q_id = line_info[0]
                question = line_info[1]
                a_id = line_info[4]
                answer = line_info[5]
                question = ' '.join(map(lambda x: wn_lemmatizer.lemmatize(x), nltk.word_tokenize(question)))
                answer = ' '.join(map(lambda x: wn_lemmatizer.lemmatize(x), nltk.word_tokenize(answer)))
                if set_name != 'test':
                    label = line_info[6]
                    fout.write('\t'.join([q_id, question, a_id, answer, label]) + '\n')
                else:
                    fout.write('\t'.join([q_id, question, a_id, answer]) + '\n')
