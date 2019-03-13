
import nltk
import csv
import pandas as pd


def load_vocabs(data_file):
    s = set()
    t = set()
    print('start read vocabulary from {} file...'.format(data_file))
    df = pd.read_csv(data_file, sep='\t', encoding='utf8')
    for _, row in df.iterrows():
        s.update([c for c in row['Pronoun']])
        s.update([c for c in row['Coref']])
        s.update([c for c in row['Pronoun-context']])
        s.update([c for c in row['Coref-context']])
        t.update([w for w in nltk.word_tokenize(row['Pronoun-context']) if w.isalpha()])
        t.update([w for w in nltk.word_tokenize(row['Coref-context']) if w.isalpha()])
    return (s, t)

s = set()
t = set()
for f in ['validation', 'dev', 'test']:
    (s1, t1) = load_vocabs('./data/{}.tsv'.format(f))
    s = s.union(s1)
    t = t.union(t1)


def write_vocab(v, vocab_file):
    print('write {} of elements to file {}'.format(len(v), vocab_file))
    with open(vocab_file, 'wb') as f:
        for r in sorted(list(v)):
            f.write('{}\n'.format(r).encode())

write_vocab(s, './data/vocab.chars.txt')
write_vocab(t, './data/vocab.words.txt')

