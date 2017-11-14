import json
from argparse import ArgumentParser
import os

def get_ner(question, answer):
    question_ner, answer_ner = [], []

    json_dict_q = json.load(question)
    json_dict_a = json.load(answer)

    for sent in json_dict_q['sentences']:
        this_ner = []
        for token in sent['tokens']:
            this_ner.append(token['ner'])
        question_ner.append(' '.join(this_ner))

    for sent in json_dict_a['sentences']:
        this_ner = []
        for token in sent['tokens']:
            this_ner.append(token['ner'])
        answer_ner.append(' '.join(this_ner))

    return '\n'.join([q + '\t' + a for q, a in zip(question_ner, answer_ner)])

if __name__ == '__main__':
    parser = ArgumentParser(description="Parse jsons")
    parser.add_argument('--input', type=str, help="directory path of the TrecQA|WikiQA")
    args = parser.parse_args()

    input = os.path.splitext(args.input)[0]
    base_dir = args.input
    if 'TrecQA' in base_dir:
        sub_dirs = ['train-all/', 'raw-dev/', 'raw-test/']
    elif 'WikiQA' in base_dir:
        sub_dirs = ['train/', 'dev/', 'test/']
    else:
        print('Unsupported dataset')
        exit()

    for sub in sub_dirs:
        this_dir = base_dir + sub
        with open(this_dir + 'a.toks.ner.json') as q_ner, \
                open(this_dir + 'b.toks.ner.json') as a_ner, \
                open(this_dir + 'ner_feats', 'w') as h:

                json2str = get_ner(q_ner, a_ner)

                h.write(json2str)
