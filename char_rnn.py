
import functools
import json
import logging
import os
import sys
import nltk
import csv
import errno
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


DATADIR = './data'
RESULTSDIR = './char_rnn_results'


Path(RESULTSDIR).mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler(RESULTSDIR + '/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def parse_fn(pronoun, pronoun_context, coref, coref_context, label):
    def get_words(text):
        return [w for w in nltk.word_tokenize(text) if max([c.isalpha() or c.isdigit() for c in w])]

    def get_chars(words):
        chars = [[c for c in w] for w in words]
        return [chars, [len(c) for c in chars]]

    def pad(chars, max_len):
        lengths = [len(c) for c in chars]
        return [c + ['<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]

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

    return ((p, p_len, len(p)), (pc, pc_len, len(pc)), (c, c_len, len(c)), (cc, cc_len, len(cc))), (s)


def generator_fn(data_file):
    with Path(data_file).open('rt', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quotechar='"')
        for row in reader:
            yield parse_fn(row['Pronoun'], row['Pronoun-context'], row['Coref'], row['Coref-context'], row['Score'])


def input_fn(data_file, params=None):
    params = params if params is not None else {}
    shapes = ((
        ([None, None], [None], ()),
        ([None, None], [None], ()),
        ([None, None], [None], ()),
        ([None, None], [None], ())),
        (()))
    types = ((
        (tf.string, tf.int32, tf.int32),
        (tf.string, tf.int32, tf.int32),
        (tf.string, tf.int32, tf.int32),
        (tf.string, tf.int32, tf.int32)),
        (tf.int32))
    defaults = ((
        ('<pad>', 0, 0),
        ('<pad>', 0, 0),
        ('<pad>', 0, 0),
        ('<pad>', 0, 0)),
        (0))

    dataset = tf.data.Dataset.from_generator(functools.partial(generator_fn, data_file), output_shapes=shapes, output_types=types)
    dataset = dataset.repeat(params['epochs'])
    return (dataset.padded_batch(params.get('batch_size', 50), shapes, defaults).prefetch(1))

def model_fn(features, labels, mode, params):
    # read inputs
    (p, p_clen, p_wlen), (pc, pc_clen, pc_wlen), (c, c_clen, c_wlen), (cc, cc_clen, cc_wlen) = features

    # read vocabs
    vocab_chars = tf.contrib.lookup.index_table_from_file(vocabulary_file=params['char_vocab_file'], num_oov_buckets=params['num_oov_buckets'])
    with open(params['char_vocab_file']) as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']

    # prepare embedding tables for chars and words
    char_embedding_table = tf.Variable(tf.random_uniform([num_chars, params['char_embedding_size']]))

    # prepare data
    p_embedding = tf.nn.embedding_lookup(char_embedding_table, vocab_chars.lookup(p))
    pc_embedding = tf.nn.embedding_lookup(char_embedding_table, vocab_chars.lookup(pc))
    c_embedding = tf.nn.embedding_lookup(char_embedding_table, vocab_chars.lookup(c))
    cc_embedding = tf.nn.embedding_lookup(char_embedding_table, vocab_chars.lookup(cc))

    # multilayer bidirectional char rnn
    def create_char_rnn(data, data_len, scope_name):
        with tf.variable_scope(scope_name):
            # put the time dimension on axis=1
            s = tf.shape(data)
            data = tf.reshape(data, shape=[s[0]*s[1], s[-2], params['char_embedding_size']])
            data_len = tf.reshape(data_len, shape=[s[0]*s[1]])

            # create bidirectional rnn
            cell_fw = tf.contrib.rnn.LSTMCell(params['char_dim'], state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(params['char_dim'], state_is_tuple=True)
            _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, data, sequence_length=data_len, dtype=tf.float32)

            # prepare state
            ((_, state_fw), (_, state_bw)) = states
            output = tf.concat([state_fw, state_bw], axis=-1)

        return tf.reshape(output, shape=[s[0], s[1], 2*params['char_dim']])

    # char rnns
    p_chars = create_char_rnn(p_embedding, p_clen, 'p-char-rnn')
    pc_chars = create_char_rnn(pc_embedding, pc_clen, 'pc-char-rnn')
    c_chars = create_char_rnn(c_embedding, c_clen, 'c-char-rnn')
    cc_chars = create_char_rnn(cc_embedding, cc_clen, 'cc-char-rnn')

    def create_word_rnn(data, data_len, scope_name):
        with tf.variable_scope(scope_name):
            cell_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(params['word_dim']) for _ in range(params['word_layers'])])
            cell_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(params['word_dim']) for _ in range(params['word_layers'])])
            _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, data, sequence_length=data_len, dtype=tf.float32)

            # prepare state
            state_fw, state_bw = states
            cells = []
            for fw, bw in zip(state_fw, state_bw):
                state = tf.concat([fw, bw], axis=-1)
                cells += [tf.layers.dense(state, params['word_dim'])]

        return tf.concat(cells, axis=-1)

    p_state = create_word_rnn(p_chars, p_wlen, 'p-word-rnn')
    pc_state = create_word_rnn(pc_chars, pc_wlen, 'pc-word-rnn')
    c_state = create_word_rnn(c_chars, c_wlen, 'c-word-rnn')
    cc_state = create_word_rnn(cc_chars, cc_wlen, 'cc-word-rnn')

    p_vector = tf.layers.dense(tf.concat([p_state, pc_state], axis=-1), params['dence_dim'])
    c_vector = tf.layers.dense(tf.concat([c_state, cc_state], axis=-1), params['dence_dim'])
    #prediction = tf.reduce_sum(tf.squared_difference(p_vector, c_vector), axis=-1)
    dense = tf.layers.dense(tf.concat([p_vector, c_vector], axis=-1), params['dence_dim'])
    logits = tf.layers.dense(dense, 2)

    #print_op = tf.print(tf.shape(prediction), prediction, tf.greater_equal(prediction, .5))
    predictions = {
        "labels": tf.argmax(input=logits, axis=1),
        #"labels": tf.greater_equal(prediction, .5),
        "distance": tf.nn.softmax(logits, name="softmax_tensor"),
        #"distance": prediction
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #with tf.control_dependencies([print_op]):
    #loss = tf.nn.l2_loss(prediction - tf.cast(labels, tf.float32))
    #loss = tf.losses.mean_squared_error(predictions=prediction, labels=tf.cast(labels, tf.float32))
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params.get('lr', .001))
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "acc": tf.metrics.accuracy(labels=labels, predictions=predictions['labels'])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
    # Params
    params = {
        'char_dim': 64,
        'word_dim': 128,
        'dence_dim': 512,
        'lr': .001,
        'clip': .5,
        'char_embedding_size': 128,
        'word_embedding_size': 256,
        'max_iters': 50,
        'dropout': 0.4,
        'char_layers': 3,
        'word_layers': 5,
        'num_oov_buckets': 3,
        'epochs': 500,
        'batch_size': 32,
        'char_vocab_file': str(Path(DATADIR, 'vocab.chars.txt')),
        'word_vocab_file': str(Path(DATADIR, 'vocab.words.txt')),
    }
    with Path(RESULTSDIR, 'params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)


    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, str(Path(DATADIR, 'dev.tsv')), params)
    eval_inpf = functools.partial(input_fn, str(Path(DATADIR, 'validation.tsv')))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, str(Path(RESULTSDIR, 'model')), cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, 'acc', 1000, min_steps=40000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        path = str(Path(DATADIR, '{}.tsv'.format(name)))
        print('\n\n------------- start prediction on {}...\n'.format(path))
        test_inpf = functools.partial(input_fn, path)
        golds_gen = generator_fn(path)
        preds_gen = estimator.predict(test_inpf)
        err = 0
        alls = 0
        for golds, preds in zip(golds_gen, preds_gen):
            (_, (target)) = golds
            alls += 1
            if preds['labels'] != target:
                err += 1
        print('alls: ', alls)
        print('errs: ', err)
        print('acc: ', 1. - (float(err) / alls))

    for name in ['test', 'validation']:
        write_predictions(name)

