from nltk.parse.stanford import StanfordDependencyParser

import string
import argparse
import os
import sys
import re

path_to_jar = 'stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar'
path_to_models_jar = 'stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
r = re.compile(r'\d{3,100}[ -]+\d{3,100}[ -]+\d{3,100}')

def dependency_parse(sentence):
    try:
        result = dependency_parser.raw_parse(sentence)
        dep = result.__next__()
        return list(dep.triples())
    except:
        print("EXCEPTION: while parsing sentence:")
        print(sentence)
        sys.exit(0)
    # return []


def get_dependency_reordered(sentence):
    sentence = sentence.replace("/", "")
    results = r.findall(sentence)

    for res in results:
        replaced = res.replace('-', "_")
        replaced = res.replace(' ', "_")
        print(replaced)
        sentence = sentence.replace(res, replaced)

    deps = dependency_parse(sentence)
    reordered_tokens = []
    for item in deps:
        reordered_tokens.append(item[0][0])
        reordered_tokens.append(item[2][0])
    return ' '.join(reordered_tokens)


def write_out_deps(filename, data):
    with open(filename, "w") as outf:
        oldq = None
        oldqdep = None
        for q in data:
            if oldq != q:
                oldqdep = get_dependency_reordered(q)
                oldq = q
            print(oldqdep if len(oldqdep) else "a_placeholder_term", file=outf)


if __name__ == "__main__":

    get_dependency_reordered(" ".join(input().split()[:60]))
    ap = argparse.ArgumentParser(description='dependecy parseing and reordering of sentence tokens')
    ap.add_argument("input_file")
    #
    ap.add_argument("offset", type=int)
    ap.add_argument("batch_size", type=int)
    args = ap.parse_args()
    #
    questions = [line.strip() for line in open(args.input_file).readlines()][args.offset:args.offset+args.batch_size]
    write_out_deps(args.input_file + ".deps.{:06d}".format(args.offset), questions)
   

