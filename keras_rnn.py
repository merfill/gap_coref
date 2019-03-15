
import functools
import json
import logging
import sys
import os
import nltk
import csv
import errno
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

import keras.backend as K
from keras.layers import Dense, Input, GRU, Bidirectional, Activation, Masking
from keras.models import Model


def parse_fn(pronoun, pronoun_context, coref, coref_context, label, d):
    # word lists
    def get_words(text):
        return [w for w in nltk.word_tokenize(text) if max([c.isalpha() or c.isdigit() for c in w])]

    # char lists of lists
    def get_chars(words):
        chars = []
        for w in words:
            w = w.strip()
            wc = []
            for c in w:
                if c not in d:
                    d[c] = len(d)+1
                wc.append(d[c])
            chars += [wc]
        return chars, [len(c) for c in chars]

    def pad(chars, max_len):
        lengths = [len(c) for c in chars]
        return [c + [0] * (max_len - l) for c, l in zip(chars, lengths)]

    p, p_len = get_chars(get_words(pronoun))
    pc, pc_len = get_chars(get_words(pronoun_context))
    c, c_len = get_chars(get_words(coref))
    cc, cc_len = get_chars(get_words(coref_context))

    max_len = max([max(p_len), max(pc_len), max(c_len), max(cc_len)])
    p = pad(p, max_len)
    pc = pad(pc, max_len)
    c = pad(c, max_len)
    cc = pad(cc, max_len)
    s = 0 if label == 'False' else 1

    return (p, pc, c, cc), (s)


def generator_fn(data_file, vocab_file):
    with Path(data_file).open('rt', encoding='utf-8') as f, Path(vocab_file).open('rt', encoding='utf-8') as v:
        d = dict()
        for line in v:
            c = line.rstrip('\n')
            if c not in d:
                d[c] = len(d) + 1
                print(c, ' ', d[c])

        reader = csv.DictReader(f, delimiter='\t', quotechar='"')
        for row in reader:
            yield parse_fn(row['Pronoun'], row['Pronoun-context'], row['Coref'], row['Coref-context'], row['Score'], d)


def input_fn(data_file, params):
    shapes = ((([None, None]),   # pronoun
                ([None, None]),  # pronoun context
                ([None, None]),  # coref
                ([None, None])), # coref context
                (()))
    types = (((tf.int32), (tf.int32), (tf.int32), (tf.int32)), (tf.int32))
    defaults = (((0),
                 (0),
                 (0),
                 (0)),
                 (0))

    dataset = tf.data.Dataset.from_generator(functools.partial(generator_fn, data_file, params['char_vocab_file']), output_shapes=shapes, output_types=types)
    return (dataset.padded_batch(params.get('batch_size', 32), shapes, defaults).prefetch(1))


def model(data, params):
    (p, pc, c, cc), s  = data.get_next()

    # read vocabs
    with open(params['char_vocab_file']) as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']

    s = Input(tensor=s)
    p = Input(tensor=p)
    pp = Input(tensor=pp)
    c = Input(tensor=c)
    cc = Input(tensor=cc)

    p_x = Embedding(num_chars, params['char_embedding_size'], mask_zero=True)(p)
    pp_x = Embedding(num_chars, params['char_embedding_size'], mask_zero=True)(pp)
    c_x = Embedding(num_chars, params['char_embedding_size'], mask_zero=True)(c)
    cc_x = Embedding(num_chars, params['char_embedding_size'], mask_zero=True)(cc)

    def rnn(x):
        for _ in range(params['char_layers']):
            mask = Masking(mask_value=0.)(x)
            gru = GRU(params['char_dim'], activation='relu', return_state=True, return_sequences=False)
            x = Bidirectional(gru, merge_mode='sum')(mask)
        return x

    p_rnn = rnn(p_x)
    pp_rnn = rnn(pp_x)
    c_rnn = rnn(c_x)
    cc_rnn = rnn(cc_x)

    concat = tf.keras.layers.Concatenate([p_rnn, pp_rnn, c_rnn, cc_rnn], axis=-1)
    dense1 = Dense(params['dence_dim'])(concat)
    dense2 = Dense(2)(dense1)
    output = Activation('softmax')(dense2)

    return Model(inputs=[s, p, pp, c, cc], outputs=[output])

DATADIR = './data'

# Params
params = {
    'char_dim': 64,
    'word_dim': 256,
    'dence_dim': 1024,
    'lr': .001,
    'clip': .5,
    'char_embedding_size': 128,
    'word_embedding_size': 256,
    'max_iters': 50,
    'dropout': 0.4,
    'word_layers': 5,
    'char_layers': 3,
    'num_oov_buckets': 3,
    'epochs': 5,
    'batch_size': 32,
    'char_vocab_file': os.path.join(DATADIR, 'vocab.chars.txt'),
    #'word_vocab_file': os.path.join(DATADIR, 'vocab.words.txt'),
}

tf.enable_eager_execution()
inpf = functools.partial(input_fn, 'data/dev.tsv', params)
model 

for epoch in range(params['epochs']):
    print('epoch: ', epoch)
    data = inpf().make_one_shot_iterator()
