import torch
import numpy as np

dep_tags = ['<unk>', '<pad>', 'csubj', 'aux', 'acl:relcl', 'mark', 'expl', 'amod', 'acl', 'parataxis', 'compound',
            'advmod', 'nmod:poss', 'cc:preconj', 'det', 'case', 'ROOT', 'punct', 'nmod:npmod',
            'nsubjpass', 'det:predet', 'advcl', 'root', 'dep', 'mwe', 'xcomp', 'nmod', 'cop',
            'cc', 'nsubj', 'csubjpass', 'appos', 'conj', 'nummod', 'discourse', 'auxpass', 'ccomp',
            'nmod:tmod', 'iobj', 'compound:prt', 'dobj', 'neg', 'NO_DEP']

pos_tags = ['<unk>', '<pad>', 'RBS', "''", 'VB', '#', '.', 'WP$', 'SYM', 'LS', 'WDT', 'NNP', 'TO', 'CD', 'NNPS',
            'NN', 'MD', 'RBR', 'JJS', 'VBN', 'VBP', '``', 'WRB', 'JJR', 'VBD', 'FW', 'RB', 'NNS',
            'POS', ',', 'PDT', 'UH', 'VBG', '$', 'PRP$', 'VBZ', 'PRP', ':', 'WP', 'IN', 'CC', 'DT',
            'JJ', 'RP', 'EX', 'NO_POS']

def one_hot(tag, index):
    tag_one_hot = np.zeros(len(tag), dtype=float)
    tag_one_hot[index] = 1.0
    return tag_one_hot

def create_lookup():
    pos_pt = './data/pos.trecqa.pt'
    pos_dim = len(pos_tags)
    stoi = {word: i for i, word in enumerate(pos_tags)}
    vectors = [one_hot(pos_tags, pos_tags.index(tag)) for tag in pos_tags]
    vectors = torch.Tensor(vectors).view(-1, pos_dim)
    print('saving vectors to', pos_pt)
    torch.save((stoi, vectors, pos_dim), pos_pt)

    dep_pt = './data/dep.trecqa.pt'
    dep_dim = len(dep_tags)
    stoi = {word: i for i, word in enumerate(dep_tags)}
    vectors = [one_hot(dep_tags, dep_tags.index(tag)) for tag in dep_tags]
    vectors = torch.Tensor(vectors).view(-1, dep_dim)
    print('saving vectors to', dep_pt)
    torch.save((stoi, vectors, dep_dim), dep_pt)

if __name__ == '__main__':
    create_lookup()