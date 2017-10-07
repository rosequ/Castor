import json
from argparse import ArgumentParser
import os

def get_dep_pos(string):
  this_pos = []
  json_dict = json.loads(string)
  this_token = []

  for token in json_dict['tokens']:
    this_token.append(token['originalText'])
    this_pos.append(token['pos'])

  len_dep = len(this_pos)
  this_dep = [None] * len_dep
  headwords = [None] * len_dep
  this_head_index = [None] * len_dep

  for dep in json_dict['basicDependencies']:
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

  return (headwords, this_head_pos, this_head_dep, this_pos, this_dep)

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
                question = question.strip()
                answer = answer.strip()
                question_details = '\t'.join([' '.join(y) for y in get_dep_pos(question)])
                answer_details = '\t'.join([' '.join(y) for y in get_dep_pos(answer)])

                h.write('{}\t{}'.format(question_details, answer_details) + '\n')
