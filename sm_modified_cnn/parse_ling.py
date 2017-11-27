import json
from argparse import ArgumentParser
import os

def get_ling(file_name):
    final_str = []
    json_dict = json.load(file_name)
    for sent in json_dict['sentences']:
        this_ner, this_pos, this_token = [], [], []
        for token in sent['tokens']:
            this_token.append(token['originalText'])
            this_pos.append(token['pos'])
            this_ner.append(token['ner'])

        len_dep = len(this_pos)
        this_dep = [None] * len_dep
        headwords = [None] * len_dep
        this_head_index = [None] * len_dep

        for dep in sent['basicDependencies']:
            index, dependency = dep['dependent'], dep['dep']
            this_dep[index - 1] = dependency
            headwords[index - 1] = dep['governorGloss']
            this_head_index[index - 1] = dep['governor']

        this_head_dep = []
        this_head_pos = []

        for head_index in this_head_index:
            if head_index - 1 < 0:
                this_head_dep.append('NO_DEP')
                this_head_pos.append('NO_POS')
            else:
                this_head_dep.append(this_dep[head_index - 1])
                this_head_pos.append(this_pos[head_index - 1])

            if (headwords[0].strip() == ''):
                print("empty string")

                #     if not (len(headwords) == len(this_head_pos) == len(this_head_dep) == len(this_pos) == len(this_dep)):
                #         print("unequal lengths")
                #         exit()

        final_str.append(' '.join(headwords) + '\t' + ' '.join(this_head_pos) + '\t' +
                         ' '.join(this_head_dep) + '\t' + ' '.join(this_pos) + '\t' +
                         ' '.join(this_dep) + '\t' + ' '.join(this_ner))
    # return '\n'.join(final_str)
    return final_str

if __name__ == '__main__':
    parser = ArgumentParser(description="Parse jsons")
    parser.add_argument('--input', type=str, help="directory path of the TrecQA|WikiQA")
    args = parser.parse_args()

    input = os.path.splitext(args.input)[0]
    base_dir = args.input
    if 'TrecQA' in base_dir:
        sub_dirs = ['train-all/', 'raw-dev/', 'raw-test/']
    elif 'cleanQA' in base_dir:
        sub_dirs = ['train-all/', 'clean-dev/', 'clean-test/']
    elif 'WikiQA' in base_dir:
        sub_dirs = ['train/', 'dev/', 'test/']
    else:
        print('Unsupported dataset')
        exit()

    for sub in sub_dirs:
        this_dir = base_dir + sub
        with open(this_dir + 'a.toks.json') as q_deps, open(this_dir + 'b.toks.json') as a_deps, \
                open(this_dir + 'dependency_feats', 'w') as h:
            for question_det, answer_det in zip(get_ling(q_deps), get_ling(a_deps)):
                h.write('{}\t{}'.format(question_det, answer_det) + '\n')
