import json
from argparse import ArgumentParser
import os

all_pos_tags = set()
all_dep_tags = set()

def get_dep_pos(string):
  json_dict = json.loads(string)

  for token in json_dict['tokens']:
    if token['pos'] not in all_pos_tags:
        all_pos_tags.add(token['pos'])

  for dep in json_dict['basicDependencies']:
    if dep['dep'] not in all_dep_tags:
        all_dep_tags.add(dep['dep'])

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
        with open(this_dir + 'a.toks.deps.json') as q_deps, open(this_dir + 'b.toks.deps.json') as a_deps, \
                open(this_dir + 'dependency_feats', 'w') as h:
            for question, answer in zip(q_deps, a_deps):
                get_dep_pos(question)
                get_dep_pos(answer)

    print(all_pos_tags, all_dep_tags)
    print(len(all_pos_tags))
    print(len(all_dep_tags))

