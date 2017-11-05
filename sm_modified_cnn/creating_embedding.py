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

to_universal_pos = {"<unk>":  "<unk>", "<pad>":  "<pad>", "!":  ".", "#":  ".", "$":  ".", "''":  ".", "(":  ".",
                   ")":  ".", ",":  ".", "-LRB-":  ".", "-RRB-":  ".", ".":  ".", ":  ":  ".", "?":  ".", "CC":  "CONJ",
                   "CD":  "NUM", "CD|RB":  "X", "DT":  "DET", "EX":  "DET", "FW":  "X", "IN":  "ADP", "IN|RP":  "ADP",
                   "JJ":  "ADJ", "JJR":  "ADJ", "JJRJR":  "ADJ", "JJS":  "ADJ", "JJ|RB":  "ADJ", "JJ|VBG":  "ADJ",
                   "LS":  "X", "MD":  "VERB", "NN":  "NOUN", "NNP":  "NOUN", "NNPS":  "NOUN", "NNS":  "NOUN",
                   "NN|NNS":  "NOUN","NN|SYM":  "NOUN", "NN|VBG":  "NOUN", "NP":  "NOUN", "PDT":  "DET", "POS":  "PRT",
                   "PRP":  "PRON", "PRP$":  "PRON", "PRP|VBP":  "PRON", "PRT":  "PRT", "RB":  "ADV", "RBR":  "ADV",
                   "RBS":  "ADV", "RB|RP":  "ADV", "RB|VBG":  "ADV", "RN":  "X", "RP":  "PRT", "SYM":  "X", "TO":  "PRT",
                   "UH":  "X", "VB":  "VERB", "VBD":  "VERB", "VBD|VBN":  "VERB", "VBG":  "VERB", "VBG|NN":  "VERB",
                   "VBN":  "VERB", "VBP":  "VERB", "VBP|TO":  "VERB", "VBZ":  "VERB", "VP":  "VERB", "WDT":  "DET",
                   "WH":  "X", "WP":  "PRON", "WP$":  "PRON", "WRB":  "ADV", "``":  ".", "NO_POS":"NO_POS", ":":"X"}
universal_pos = ['<unk>', '<pad>', 'VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X', '.', 'NO_POS']

def one_hot(tag, index):
    tag_one_hot = np.zeros(len(tag), dtype=float)
    tag_one_hot[index] = 1.0
    return tag_one_hot

def lookup_pos(tag, coarse):
    pos_dim = len(universal_pos) if coarse else len(pos_tags)
    vectors = None
    try:
        if coarse:
            vectors = one_hot(universal_pos, universal_pos.index(to_universal_pos[tag]))
        else:
            vectors = one_hot(pos_tags, pos_tags.index(tag))
    except KeyError as e:
        print(e)
        vectors = torch.FloatTensor(pos_dim).uniform_(-0.25, 0.25)
    vectors = torch.Tensor(vectors).view(-1, pos_dim)
    return vectors

def create_embeddings():
    pos_pt = './data/pos.trecqa.pt'
    pos_dim = len(pos_tags)
    stoi = {word: i for i, word in enumerate(pos_tags)}
    vectors = [one_hot(pos_tags, pos_tags.index(tag)) for tag in pos_tags]
    vectors = torch.Tensor(vectors).view(-1, pos_dim)
    print('saving vectors to', pos_pt)
    torch.save((stoi, vectors, pos_dim), pos_pt)

    # pos_pt = './data/universal.pos.trecqa.pt'
    # pos_dim = len(universal_pos)
    # stoi = {word: i for i, word in enumerate(universal_pos)}
    # vectors = [one_hot(universal_pos, universal_pos.index(tag)) for tag in universal_pos]
    # vectors = torch.Tensor(vectors).view(-1, pos_dim)
    # print('saving vectors to', pos_pt)
    # torch.save((stoi, vectors, pos_dim), pos_pt)

    dep_pt = './data/dep.trecqa.pt'
    dep_dim = len(dep_tags)
    stoi = {word: i for i, word in enumerate(dep_tags)}
    vectors = [one_hot(dep_tags, dep_tags.index(tag)) for tag in dep_tags]
    vectors = torch.Tensor(vectors).view(-1, dep_dim)
    print('saving vectors to', dep_pt)
    torch.save((stoi, vectors, dep_dim), dep_pt)

if __name__ == '__main__':
    create_embeddings()