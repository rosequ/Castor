import re
import json

from nltk.tokenize import TreebankWordTokenizer

dep_tags = ['csubj', 'aux', 'acl:relcl', 'mark', 'expl', 'amod', 'acl', 'parataxis', 'compound',
            'advmod', 'nmod:poss', 'cc:preconj', 'det', 'case', 'ROOT', 'punct', 'nmod:npmod',
            'nsubjpass', 'det:predet', 'advcl', 'root', 'dep', 'mwe', 'xcomp', 'nmod', 'cop',
            'cc', 'nsubj', 'csubjpass', 'appos', 'conj', 'nummod', 'discourse', 'auxpass', 'ccomp',
            'nmod:tmod', 'iobj', 'compound:prt', 'dobj', 'neg', 'NO_DEP']

pos_tags = ['RBS', "''", 'VB', '#', '.', 'WP$', 'SYM', 'LS', 'WDT', 'NNP', 'TO', 'CD', 'NNPS',
            'NN', 'MD', 'RBR', 'JJS', 'VBN', 'VBP', '``', 'WRB', 'JJR', 'VBD', 'FW', 'RB', 'NNS',
            'POS', ',', 'PDT', 'UH', 'VBG', '$', 'PRP$', 'VBZ', 'PRP', ':', 'WP', 'IN', 'CC', 'DT',
            'JJ', 'RP', 'EX', 'NO_POS']


def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " ( ", string)
  string = re.sub(r"\)", " ) ", string)
  string = re.sub(r"\?", " ? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip()


def clean_str_sst(string):
  """
  Tokenization/string cleaning for the SST dataset
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip()

  # return TreebankWordTokenizer().tokenize(string)


def get_dep_pos(string):
  this_pos = []
  json_dict = json.loads(string)

  for token in json_dict['tokens']:
    this_pos.append(pos_tags.index(token['pos']))

  len_dep = len(this_pos)
  this_dep = [None] * len_dep
  headwords = [None] * len_dep
  this_head_index = [None] * len_dep

  for dep in json_dict['basicDependencies']:
    index, dependency = dep['dependent'], dep['dep']
    this_dep[index - 1] = dep_tags.index(dependency)
    headwords[index - 1] = dep['governorGloss']
    this_head_index[index - 1] = dep['governor']

  this_head_dep = []
  this_head_pos = []

  for head_index in this_head_index:
    if head_index - 1 < 0:
      this_head_dep.append(dep_tags.index('NO_DEP'))
      this_head_pos.append(pos_tags.index('NO_POS'))
    else:
      this_head_dep.append(this_dep[head_index - 1])
      this_head_pos.append(this_pos[head_index - 1])

  tags = this_dep + this_pos
  headtags = this_head_dep + this_head_pos
  return headwords, headtags, tags
