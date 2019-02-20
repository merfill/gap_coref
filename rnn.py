
from __future__ import print_function
import functools
import json
import logging
import os
import sys
import nltk
import csv
import errno
import random

import numpy as np
import tensorflow as tf


DATADIR = './data'
RESULTSDIR = './rnn_results'


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


# Logging
tf.logging.set_verbosity(logging.INFO)

mkdir(RESULTSDIR)
handlers = [
    logging.FileHandler(RESULTSDIR + '/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def parse_fn(pronoun, pronoun_context, coref, coref_context, label):
    ps = [c.encode('utf8') for c in pronoun.decode('utf8').rstrip('\n')]
    pcs = [w.encode('utf8') for w in nltk.word_tokenize(pronoun_context.decode('utf8').strip())]
    cs = [c.encode('utf8') for c in coref.decode('utf8').rstrip('\n')]
    ccs = [w.encode('utf8') for w in nltk.word_tokenize(coref_context.decode('utf8').strip())]
    t = 0 if label == 'False' else 1
    return ((ps, len(ps)), (pcs, len(pcs)), (cs, len(cs)), (ccs, len(ccs))), (t)


def generator_fn(data_file):
    with open(data_file, 'rb') as f:
        reader = csv.DictReader(f, delimiter='\t', quotechar='"')
        for row in reader:
            yield parse_fn(row['Pronoun'], row['Pronoun-context'], row['Coref'], row['Coref-context'], row['Score'])


def input_fn(data_file, params=None):
    params = params if params is not None else {}
    shapes = ((([None], ()), ([None], ()), ([None], ()), ([None], ())), (()))
    types = (((tf.string, tf.int32), (tf.string, tf.int32), (tf.string, tf.int32), (tf.string, tf.int32)), (tf.int32))
    defaults = ((('<pad>', 0), ('<pad>', 0), ('<pad>', 0), ('<pad>', 0)), (0))

    dataset = tf.data.Dataset.from_generator(functools.partial(generator_fn, data_file), output_shapes=shapes, output_types=types)
    dataset = dataset.repeat(params['epochs'])
    return (dataset.padded_batch(params.get('batch_size', 50), shapes, defaults).prefetch(1))

def model_fn(features, labels, mode, params):
    # read inputs
    (ps, ps_len), (pcs, pcs_len), (cs, cs_len), (ccs, ccs_len)  = features

    # read vocabs
    vocab_chars = tf.contrib.lookup.index_table_from_file(vocabulary_file=params['char_vocab_file'], num_oov_buckets=params['num_oov_buckets'])
    with open(params['char_vocab_file']) as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']
    vocab_words = tf.contrib.lookup.index_table_from_file(vocabulary_file=params['word_vocab_file'], num_oov_buckets=params['num_oov_buckets'])
    with open(params['word_vocab_file']) as f:
        num_words = sum(1 for _ in f) + params['num_oov_buckets']

    # prepare embedding tables for chars and words
    char_embedding_table = tf.Variable(tf.random_uniform([num_chars, params['char_embedding_size']]))
    word_embedding_table = tf.Variable(tf.random_uniform([num_words, params['word_embedding_size']]))

    # prepare data
    ps_embedding = tf.nn.embedding_lookup(char_embedding_table, vocab_chars.lookup(ps))
    pcs_embedding = tf.nn.embedding_lookup(word_embedding_table, vocab_words.lookup(pcs))
    cs_embedding = tf.nn.embedding_lookup(char_embedding_table, vocab_chars.lookup(cs))
    ccs_embedding = tf.nn.embedding_lookup(word_embedding_table, vocab_words.lookup(ccs))

    # multilayer bidirectional rnn
    def create_rnn(data, data_len, scope_name):
        with tf.variable_scope(None, scope_name):
            # create bidirectional rnn
            cell_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(params['dim']) for _ in range(params['layers'])])
            cell_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(params['dim']) for _ in range(params['layers'])])
            _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, data, sequence_length=data_len, dtype=tf.float32)

            # prepare state
            state_fw, state_bw = states
            cells = []
            for fw, bw in zip(state_fw, state_bw):
                state = tf.concat([fw, bw], axis=-1)
                cells += [tf.layers.dense(state, params['dim'])]

        return tf.concat(cells, axis=-1)

    ps_state = create_rnn(ps_embedding, ps_len, 'ps')
    pcs_state = create_rnn(pcs_embedding, pcs_len, 'pcs')
    cs_state = create_rnn(cs_embedding, cs_len, 'cs')
    ccs_state = create_rnn(ccs_embedding, ccs_len, 'ccs')
    dense = tf.layers.dense(tf.concat([ps_state, pcs_state, cs_state, ccs_state], axis=-1), params['dence_dim'])
    dropout = tf.layers.dropout(inputs=dense, rate=params['dropout'], training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(dropout, 2)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params.get('lr', .001))
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "acc": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
    # Params
    params = {
        'dim': 256,
        'dence_dim': 1024,
        'lr': .001,
        'clip': .5,
        'char_embedding_size': 128,
        'word_embedding_size': 256,
        'max_iters': 50,
        'dropout': 0.4,
        'layers': 5,
        'num_oov_buckets': 3,
        'epochs': 5,
        'batch_size': 50,
        'char_vocab_file': os.path.join(DATADIR, 'vocab.chars.txt'),
        'word_vocab_file': os.path.join(DATADIR, 'vocab.words.txt'),
    }
    with open('{}/params.json'.format(RESULTSDIR), 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)


    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, os.path.join(DATADIR, 'dev.tsv'), params)
    eval_inpf = functools.partial(input_fn, os.path.join(DATADIR, 'val.tsv'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, RESULTSDIR + '/model', cfg, params)
    mkdir(estimator.eval_dir())
    hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, 'acc', 1000, min_steps=20000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        path = os.path.join(DATADIR, '{}.tsv'.format(name))
        print('\n\n------------- start prediction on {}...\n'.format(path))
        test_inpf = functools.partial(input_fn, path)
        golds_gen = generator_fn(path)
        preds_gen = estimator.predict(test_inpf)
        err = 0
        alls = 0
        for golds, preds in zip(golds_gen, preds_gen):
            (_, (target)) = golds
            alls += 1
            if preds['classes'] != target:
                err += 1
        print('alls: ', alls)
        print('errs: ', err)
        print('acc: ', 1. - (float(err) / alls))

    for name in ['test', 'val']:
        write_predictions(name)

