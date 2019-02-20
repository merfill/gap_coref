
import pandas as pd
from nltk.tokenize import WhitespaceTokenizer


def get_data(text, pos, width):
    span_generator = WhitespaceTokenizer().span_tokenize(text)
    state = 0
    res = []
    for l, r in span_generator:
        if state == 0:
            if l == pos:
                state = 1
            else:
                res += [''.join(text[l:r])]
                if len(res) > width:
                    res.pop(0)
        elif state == 1:
            res += [''.join(text[l:r])]
            if len(res) >= width * 2:
                break
    return ' '.join(res)


def to_file(input_path, output_path):
    res_df = pd.DataFrame(columns=['ID', 'Pronoun', 'Pronoun-context', 'Coref', 'Coref-context', 'Score'])
    df = pd.read_csv(input_path, sep='\t', encoding='utf8')
    for _, row in df.iterrows():
        res_df.loc[len(res_df)] = [row['ID'], row['Pronoun'], get_data(row['Text'], row['Pronoun-offset'], 5), row['A'], get_data(row['Text'], row['A-offset'], 5), row['A-coref']]
        res_df.loc[len(res_df)] = [row['ID'], row['Pronoun'], get_data(row['Text'], row['Pronoun-offset'], 5), row['B'], get_data(row['Text'], row['B-offset'], 5), row['B-coref']]
    res_df.to_csv(output_path, sep='\t', index=False)

to_file('data/gap-development.tsv', 'data/dev.tsv')
to_file('data/gap-test.tsv', 'data/test.tsv')
to_file('data/gap-validation.tsv', 'data/validation.tsv')
